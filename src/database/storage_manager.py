"""
Production Storage Management
File storage, backup, and data lifecycle management for production environments
"""

import os
import shutil
import gzip
import tarfile
import logging
import threading
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False


@dataclass
class StorageConfig:
    """Storage configuration settings"""
    raw_srts_dir: str
    processed_srts_dir: str
    lexicons_dir: str
    logs_dir: str
    metrics_dir: str
    backups_dir: str
    max_file_age_days: int = 365
    cleanup_schedule_hours: int = 24
    backup_retention_days: int = 30


@dataclass
class BackupConfig:
    """Backup configuration settings"""
    enabled: bool = True
    schedule_cron: str = "0 3 * * *"  # Daily at 3 AM
    retention_days: int = 30
    compress: bool = True
    remote_backup_enabled: bool = False
    remote_provider: str = "s3"
    remote_bucket: Optional[str] = None
    remote_region: Optional[str] = None


@dataclass
class StorageMetrics:
    """Storage system metrics"""
    total_files: int = 0
    total_size_bytes: int = 0
    raw_srts_count: int = 0
    processed_srts_count: int = 0
    backup_count: int = 0
    last_backup_time: Optional[datetime] = None
    storage_health: str = "healthy"


class BackupManager:
    """Production backup management with local and remote storage"""
    
    def __init__(self, config: BackupConfig, storage_config: StorageConfig):
        self.config = config
        self.storage_config = storage_config
        self.logger = logging.getLogger(__name__)
        self.s3_client = None
        
        if self.config.remote_backup_enabled and AWS_AVAILABLE:
            self._initialize_s3_client()
            
        self._backup_thread = None
        self._shutdown = False
        
        if self.config.enabled:
            self._start_backup_scheduler()
            
    def _initialize_s3_client(self):
        """Initialize S3 client for remote backups"""
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.config.remote_region
            )
            
            # Test S3 connection
            self.s3_client.head_bucket(Bucket=self.config.remote_bucket)
            self.logger.info(
                "S3 backup client initialized",
                bucket=self.config.remote_bucket,
                region=self.config.remote_region
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize S3 client", exception=e)
            self.s3_client = None
            
    def _start_backup_scheduler(self):
        """Start backup scheduler thread"""
        self._backup_thread = threading.Thread(
            target=self._backup_scheduler_loop,
            daemon=True
        )
        self._backup_thread.start()
        
    def _backup_scheduler_loop(self):
        """Backup scheduler loop"""
        while not self._shutdown:
            try:
                # Simple daily backup check (in production, use proper cron scheduling)
                current_hour = datetime.now().hour
                if current_hour == 3:  # 3 AM backup
                    if not hasattr(self, '_last_backup_date') or \
                       self._last_backup_date != datetime.now().date():
                        self.create_backup()
                        self._last_backup_date = datetime.now().date()
                        
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error("Backup scheduler error", exception=e)
                time.sleep(300)  # Wait 5 minutes on error
                
    def create_backup(self) -> str:
        """Create backup of all data directories"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"asr_backup_{timestamp}"
            backup_path = os.path.join(self.storage_config.backups_dir, f"{backup_name}.tar.gz")
            
            # Ensure backup directory exists
            os.makedirs(self.storage_config.backups_dir, exist_ok=True)
            
            # Create compressed backup
            with tarfile.open(backup_path, "w:gz" if self.config.compress else "w") as tar:
                # Backup all data directories
                directories_to_backup = [
                    self.storage_config.raw_srts_dir,
                    self.storage_config.processed_srts_dir,
                    self.storage_config.lexicons_dir,
                    self.storage_config.metrics_dir,
                ]
                
                for directory in directories_to_backup:
                    if os.path.exists(directory):
                        tar.add(directory, arcname=os.path.basename(directory))
                        self.logger.info(f"Added {directory} to backup")
                        
            # Calculate backup size and hash
            backup_size = os.path.getsize(backup_path)
            backup_hash = self._calculate_file_hash(backup_path)
            
            # Create backup metadata
            metadata = {
                'name': backup_name,
                'timestamp': timestamp,
                'size_bytes': backup_size,
                'hash': backup_hash,
                'directories_included': directories_to_backup,
                'compressed': self.config.compress
            }
            
            metadata_path = backup_path.replace('.tar.gz', '.json').replace('.tar', '.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(
                "Backup created successfully",
                backup_path=backup_path,
                size_mb=backup_size / 1024 / 1024,
                hash=backup_hash
            )
            
            # Upload to remote storage if configured
            if self.config.remote_backup_enabled and self.s3_client:
                self._upload_to_s3(backup_path, backup_name)
                self._upload_to_s3(metadata_path, f"{backup_name}_metadata")
                
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            self.logger.error("Backup creation failed", exception=e)
            raise
            
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
        
    def _upload_to_s3(self, file_path: str, s3_key: str):
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(
                file_path,
                self.config.remote_bucket,
                f"backups/{s3_key}"
            )
            self.logger.info(f"Uploaded {file_path} to S3", s3_key=s3_key)
            
        except Exception as e:
            self.logger.error("S3 upload failed", file_path=file_path, exception=e)
            
    def _cleanup_old_backups(self):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            
            # Cleanup local backups
            backup_dir = Path(self.storage_config.backups_dir)
            if backup_dir.exists():
                for backup_file in backup_dir.glob("asr_backup_*.tar*"):
                    if backup_file.stat().st_mtime < cutoff_date.timestamp():
                        backup_file.unlink()
                        # Also remove metadata file
                        metadata_file = backup_file.with_suffix('.json')
                        if metadata_file.exists():
                            metadata_file.unlink()
                        self.logger.info(f"Removed old backup: {backup_file}")
                        
            # Cleanup remote backups
            if self.config.remote_backup_enabled and self.s3_client:
                self._cleanup_s3_backups(cutoff_date)
                
        except Exception as e:
            self.logger.error("Backup cleanup failed", exception=e)
            
    def _cleanup_s3_backups(self, cutoff_date: datetime):
        """Cleanup old S3 backups"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.remote_bucket,
                Prefix="backups/"
            )
            
            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.config.remote_bucket,
                        Key=obj['Key']
                    )
                    self.logger.info(f"Removed old S3 backup: {obj['Key']}")
                    
        except Exception as e:
            self.logger.error("S3 backup cleanup failed", exception=e)
            
    def restore_backup(self, backup_name: str, restore_path: str) -> bool:
        """Restore backup to specified path"""
        try:
            backup_path = os.path.join(self.storage_config.backups_dir, f"{backup_name}.tar.gz")
            
            # Try local backup first
            if not os.path.exists(backup_path) and self.s3_client:
                # Download from S3
                self.s3_client.download_file(
                    self.config.remote_bucket,
                    f"backups/{backup_name}",
                    backup_path
                )
                
            # Extract backup
            with tarfile.open(backup_path, "r:gz" if self.config.compress else "r") as tar:
                tar.extractall(restore_path)
                
            self.logger.info(f"Backup restored successfully to {restore_path}")
            return True
            
        except Exception as e:
            self.logger.error("Backup restoration failed", exception=e)
            return False
            
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        # Local backups
        backup_dir = Path(self.storage_config.backups_dir)
        if backup_dir.exists():
            for backup_file in backup_dir.glob("asr_backup_*.tar*"):
                metadata_file = backup_file.with_suffix('.json')
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        metadata['location'] = 'local'
                        metadata['path'] = str(backup_file)
                        backups.append(metadata)
                        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
        
    def close(self):
        """Shutdown backup manager"""
        self._shutdown = True
        
        if self._backup_thread and self._backup_thread.is_alive():
            self._backup_thread.join(timeout=5)
            
        self.logger.info("Backup manager shutdown completed")


class ProductionStorageManager:
    """Production storage manager with lifecycle management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage configuration
        directories_config = self.config.get('directories', {})
        self.storage_config = StorageConfig(
            raw_srts_dir=directories_config.get('raw_srts', '/app/data/raw_srts'),
            processed_srts_dir=directories_config.get('processed_srts', '/app/data/processed_srts'),
            lexicons_dir=directories_config.get('lexicons', '/app/data/lexicons'),
            logs_dir=directories_config.get('logs', '/app/logs'),
            metrics_dir=directories_config.get('metrics', '/app/data/metrics'),
            backups_dir=directories_config.get('backups', '/app/data/backups'),
        )
        
        # Initialize backup manager
        backup_config_dict = self.config.get('backup', {})
        backup_config = BackupConfig(
            enabled=backup_config_dict.get('enabled', True),
            schedule_cron=backup_config_dict.get('schedule_cron', '0 3 * * *'),
            retention_days=backup_config_dict.get('retention_days', 30),
            compress=backup_config_dict.get('compress', True),
            remote_backup_enabled=backup_config_dict.get('remote_backup', {}).get('enabled', False),
            remote_provider=backup_config_dict.get('remote_backup', {}).get('provider', 's3'),
            remote_bucket=backup_config_dict.get('remote_backup', {}).get('bucket'),
            remote_region=backup_config_dict.get('remote_backup', {}).get('region'),
        )
        
        self.backup_manager = BackupManager(backup_config, self.storage_config)
        
        # Initialize storage directories
        self._initialize_directories()
        
        # Start cleanup scheduler
        self._cleanup_thread = None
        self._shutdown = False
        self._start_cleanup_scheduler()
        
    def _initialize_directories(self):
        """Create and initialize all storage directories"""
        directories = [
            self.storage_config.raw_srts_dir,
            self.storage_config.processed_srts_dir,
            self.storage_config.lexicons_dir,
            self.storage_config.logs_dir,
            self.storage_config.metrics_dir,
            self.storage_config.backups_dir,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Storage directory initialized: {directory}")
            
    def _start_cleanup_scheduler(self):
        """Start cleanup scheduler thread"""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_scheduler_loop,
            daemon=True
        )
        self._cleanup_thread.start()
        
    def _cleanup_scheduler_loop(self):
        """Cleanup scheduler loop"""
        while not self._shutdown:
            try:
                self._perform_cleanup()
                time.sleep(self.storage_config.cleanup_schedule_hours * 3600)
            except Exception as e:
                self.logger.error("Storage cleanup error", exception=e)
                time.sleep(3600)  # Wait 1 hour on error
                
    def _perform_cleanup(self):
        """Perform storage cleanup based on retention policies"""
        cutoff_date = datetime.now() - timedelta(days=self.storage_config.max_file_age_days)
        
        cleanup_directories = [
            self.storage_config.logs_dir,
            self.storage_config.metrics_dir,
        ]
        
        for directory in cleanup_directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.getmtime(file_path) < cutoff_date.timestamp():
                            os.remove(file_path)
                            self.logger.info(f"Cleaned up old file: {file_path}")
                            
    def store_file(self, source_path: str, storage_type: str, filename: Optional[str] = None) -> str:
        """Store file in appropriate directory"""
        storage_dirs = {
            'raw_srt': self.storage_config.raw_srts_dir,
            'processed_srt': self.storage_config.processed_srts_dir,
            'lexicon': self.storage_config.lexicons_dir,
            'metrics': self.storage_config.metrics_dir,
        }
        
        if storage_type not in storage_dirs:
            raise ValueError(f"Invalid storage type: {storage_type}")
            
        target_dir = storage_dirs[storage_type]
        target_filename = filename or os.path.basename(source_path)
        target_path = os.path.join(target_dir, target_filename)
        
        # Copy file to storage
        shutil.copy2(source_path, target_path)
        
        self.logger.info(
            "File stored successfully",
            source=source_path,
            target=target_path,
            type=storage_type
        )
        
        return target_path
        
    def retrieve_file(self, filename: str, storage_type: str) -> Optional[str]:
        """Retrieve file from storage"""
        storage_dirs = {
            'raw_srt': self.storage_config.raw_srts_dir,
            'processed_srt': self.storage_config.processed_srts_dir,
            'lexicon': self.storage_config.lexicons_dir,
            'metrics': self.storage_config.metrics_dir,
        }
        
        if storage_type not in storage_dirs:
            return None
            
        file_path = os.path.join(storage_dirs[storage_type], filename)
        return file_path if os.path.exists(file_path) else None
        
    def list_files(self, storage_type: str, pattern: str = "*") -> List[str]:
        """List files in storage directory"""
        storage_dirs = {
            'raw_srt': self.storage_config.raw_srts_dir,
            'processed_srt': self.storage_config.processed_srts_dir,
            'lexicon': self.storage_config.lexicons_dir,
            'metrics': self.storage_config.metrics_dir,
        }
        
        if storage_type not in storage_dirs:
            return []
            
        directory = Path(storage_dirs[storage_type])
        if directory.exists():
            return [str(f) for f in directory.glob(pattern)]
        return []
        
    def get_storage_metrics(self) -> StorageMetrics:
        """Get comprehensive storage metrics"""
        metrics = StorageMetrics()
        
        try:
            # Count files and calculate sizes
            for storage_type, directory in {
                'raw_srt': self.storage_config.raw_srts_dir,
                'processed_srt': self.storage_config.processed_srts_dir,
            }.items():
                if os.path.exists(directory):
                    files = list(Path(directory).glob("*.srt"))
                    if storage_type == 'raw_srt':
                        metrics.raw_srts_count = len(files)
                    else:
                        metrics.processed_srts_count = len(files)
                        
                    for file_path in files:
                        metrics.total_size_bytes += file_path.stat().st_size
                        metrics.total_files += 1
                        
            # Backup metrics
            backup_dir = Path(self.storage_config.backups_dir)
            if backup_dir.exists():
                backup_files = list(backup_dir.glob("asr_backup_*.tar*"))
                metrics.backup_count = len(backup_files)
                
                if backup_files:
                    latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
                    metrics.last_backup_time = datetime.fromtimestamp(latest_backup.stat().st_mtime)
                    
        except Exception as e:
            self.logger.error("Failed to collect storage metrics", exception=e)
            metrics.storage_health = "error"
            
        return metrics
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get storage system health status"""
        status = {
            'overall_health': 'healthy',
            'directories': {},
            'backup_system': {},
            'metrics': {}
        }
        
        # Check directory health
        directories = {
            'raw_srts': self.storage_config.raw_srts_dir,
            'processed_srts': self.storage_config.processed_srts_dir,
            'lexicons': self.storage_config.lexicons_dir,
            'logs': self.storage_config.logs_dir,
            'metrics': self.storage_config.metrics_dir,
            'backups': self.storage_config.backups_dir,
        }
        
        for name, directory in directories.items():
            try:
                # Check if directory exists and is writable
                exists = os.path.exists(directory)
                writable = os.access(directory, os.W_OK) if exists else False
                
                status['directories'][name] = {
                    'exists': exists,
                    'writable': writable,
                    'path': directory,
                    'status': 'healthy' if exists and writable else 'error'
                }
                
                if not (exists and writable):
                    status['overall_health'] = 'degraded'
                    
            except Exception as e:
                status['directories'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                status['overall_health'] = 'degraded'
                
        # Get storage metrics
        try:
            storage_metrics = self.get_storage_metrics()
            status['metrics'] = {
                'total_files': storage_metrics.total_files,
                'total_size_bytes': storage_metrics.total_size_bytes,
                'raw_srts_count': storage_metrics.raw_srts_count,
                'processed_srts_count': storage_metrics.processed_srts_count,
                'backup_count': storage_metrics.backup_count,
                'last_backup': storage_metrics.last_backup_time.isoformat() if storage_metrics.last_backup_time else None,
            }
        except Exception as e:
            status['metrics'] = {'error': str(e)}
            
        return status
        
    def close(self):
        """Shutdown storage manager"""
        self._shutdown = True
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
            
        self.backup_manager.close()
        self.logger.info("Storage manager shutdown completed")


# Global storage manager instance
_storage_manager = None


def initialize_storage_production(config: Dict[str, Any]) -> ProductionStorageManager:
    """Initialize production storage manager"""
    global _storage_manager
    _storage_manager = ProductionStorageManager(config)
    return _storage_manager


def get_storage_manager() -> Optional[ProductionStorageManager]:
    """Get the global storage manager instance"""
    return _storage_manager