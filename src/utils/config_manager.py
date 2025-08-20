"""
Centralized configuration management system.

This module provides standardized configuration loading, validation, and management
across all components with support for environment-specific configurations,
secure handling, and runtime configuration updates.

Author: Dev Agent (Story 5.3)  
Version: 1.0
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import hashlib

from .exception_hierarchy import ConfigurationError, ValidationError


class ConfigurationFormat(Enum):
    """Supported configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"


class ConfigurationEnvironment(Enum):
    """Configuration environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ConfigurationSchema:
    """Configuration validation schema definition."""
    required_keys: List[str] = field(default_factory=list)
    optional_keys: List[str] = field(default_factory=list)
    key_types: Dict[str, type] = field(default_factory=dict)
    key_validators: Dict[str, Callable] = field(default_factory=dict)
    nested_schemas: Dict[str, 'ConfigurationSchema'] = field(default_factory=dict)


@dataclass
class ConfigurationSource:
    """Configuration source information."""
    path: Path
    format: ConfigurationFormat
    environment: Optional[ConfigurationEnvironment] = None
    priority: int = 0
    watch_for_changes: bool = False
    last_modified: Optional[float] = None
    checksum: Optional[str] = None


class ConfigurationManager:
    """
    Centralized configuration management system.
    
    Provides unified configuration loading, validation, environment management,
    and runtime configuration updates with proper error handling and security.
    """
    
    def __init__(self, base_config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            base_config_dir: Base directory for configuration files
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration storage
        self.config: Dict[str, Any] = {}
        self.sources: List[ConfigurationSource] = []
        self.schemas: Dict[str, ConfigurationSchema] = {}
        
        # Environment and paths
        self.base_config_dir = Path(base_config_dir or "config")
        self.environment = self._detect_environment()
        
        # Security and validation
        self.secure_keys = set()  # Keys that should not be logged
        self.validation_enabled = True
        
        # Change monitoring
        self.change_callbacks: List[Callable] = []
        self.watch_thread: Optional[threading.Thread] = None
        self.watching = False
        self._lock = threading.Lock()
        
        # Default configuration schemas
        self._register_default_schemas()
        
        self.logger.info(f"ConfigurationManager initialized for {self.environment.value} environment")
    
    def _detect_environment(self) -> ConfigurationEnvironment:
        """Detect current environment from environment variables."""
        env_var = os.getenv('APP_ENV', os.getenv('ENVIRONMENT', 'development')).lower()
        
        env_mapping = {
            'dev': ConfigurationEnvironment.DEVELOPMENT,
            'development': ConfigurationEnvironment.DEVELOPMENT,
            'test': ConfigurationEnvironment.TESTING,
            'testing': ConfigurationEnvironment.TESTING,
            'stage': ConfigurationEnvironment.STAGING,
            'staging': ConfigurationEnvironment.STAGING,
            'prod': ConfigurationEnvironment.PRODUCTION,
            'production': ConfigurationEnvironment.PRODUCTION
        }
        
        return env_mapping.get(env_var, ConfigurationEnvironment.DEVELOPMENT)
    
    def _register_default_schemas(self):
        """Register default configuration schemas for common components."""
        
        # Main application schema
        self.register_schema('app', ConfigurationSchema(
            required_keys=['name', 'version'],
            optional_keys=['description', 'debug_mode'],
            key_types={
                'name': str,
                'version': str,
                'description': str,
                'debug_mode': bool
            }
        ))
        
        # Logging configuration schema
        self.register_schema('logging', ConfigurationSchema(
            required_keys=['level'],
            optional_keys=['format', 'file_path', 'max_file_size', 'backup_count'],
            key_types={
                'level': str,
                'format': str,
                'file_path': str,
                'max_file_size': int,
                'backup_count': int
            },
            key_validators={
                'level': lambda x: x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            }
        ))
        
        # Database/data source configuration schema
        self.register_schema('data_sources', ConfigurationSchema(
            required_keys=[],
            optional_keys=['raw_srts_path', 'processed_srts_path', 'lexicons_path'],
            key_types={
                'raw_srts_path': str,
                'processed_srts_path': str,
                'lexicons_path': str
            },
            key_validators={
                'raw_srts_path': lambda x: Path(x).exists() if x else True,
                'processed_srts_path': lambda x: Path(x).parent.exists() if x else True
            }
        ))
        
        # Processing configuration schema
        self.register_schema('processing', ConfigurationSchema(
            required_keys=[],
            optional_keys=[
                'enable_sanskrit_processing', 'enable_ner', 'enable_mcp_processing',
                'performance_monitoring', 'batch_size', 'max_processing_time_ms'
            ],
            key_types={
                'enable_sanskrit_processing': bool,
                'enable_ner': bool, 
                'enable_mcp_processing': bool,
                'performance_monitoring': bool,
                'batch_size': int,
                'max_processing_time_ms': float
            },
            key_validators={
                'batch_size': lambda x: x > 0 and x <= 1000,
                'max_processing_time_ms': lambda x: x > 0
            }
        ))
        
        # MCP configuration schema
        self.register_schema('mcp', ConfigurationSchema(
            required_keys=[],
            optional_keys=['server_url', 'timeout_seconds', 'retry_attempts', 'cache_enabled'],
            key_types={
                'server_url': str,
                'timeout_seconds': float,
                'retry_attempts': int,
                'cache_enabled': bool
            },
            key_validators={
                'timeout_seconds': lambda x: x > 0 and x <= 300,
                'retry_attempts': lambda x: x >= 0 and x <= 10
            }
        ))
        
        # Mark secure keys
        self.secure_keys.update(['password', 'api_key', 'secret', 'token'])
    
    def register_schema(self, config_key: str, schema: ConfigurationSchema):
        """Register a configuration schema for validation."""
        self.schemas[config_key] = schema
        self.logger.debug(f"Registered schema for {config_key}")
    
    def add_configuration_source(
        self,
        source_path: Union[str, Path],
        format: Optional[ConfigurationFormat] = None,
        priority: int = 0,
        environment: Optional[ConfigurationEnvironment] = None,
        watch_for_changes: bool = False
    ):
        """
        Add a configuration source.
        
        Args:
            source_path: Path to configuration file
            format: Configuration format (auto-detected if None)
            priority: Source priority (higher = more important)
            environment: Target environment (None = all environments)
            watch_for_changes: Whether to watch for file changes
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise ConfigurationError(
                f"Configuration source not found: {source_path}",
                config_file=str(source_path)
            )
        
        # Auto-detect format if not specified
        if format is None:
            extension = source_path.suffix.lower()
            if extension in ['.yaml', '.yml']:
                format = ConfigurationFormat.YAML
            elif extension == '.json':
                format = ConfigurationFormat.JSON
            elif extension == '.env':
                format = ConfigurationFormat.ENV
            else:
                raise ConfigurationError(
                    f"Unable to detect configuration format for {source_path}",
                    config_file=str(source_path)
                )
        
        # Calculate file checksum for change detection
        checksum = self._calculate_file_checksum(source_path)
        
        source = ConfigurationSource(
            path=source_path,
            format=format,
            environment=environment,
            priority=priority,
            watch_for_changes=watch_for_changes,
            last_modified=source_path.stat().st_mtime,
            checksum=checksum
        )
        
        self.sources.append(source)
        self.sources.sort(key=lambda x: x.priority, reverse=True)  # Higher priority first
        
        self.logger.info(f"Added configuration source: {source_path} (priority: {priority})")
    
    def load_configuration(self, reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from all sources.
        
        Args:
            reload: Whether to reload even if already loaded
            
        Returns:
            Loaded configuration dictionary
        """
        if not reload and self.config:
            return self.config
        
        with self._lock:
            new_config = {}
            
            for source in self.sources:
                # Skip if source is environment-specific and doesn't match
                if source.environment and source.environment != self.environment:
                    continue
                
                try:
                    source_config = self._load_source(source)
                    
                    # Merge configuration (higher priority sources override)
                    new_config = self._merge_configurations(new_config, source_config)
                    
                    self.logger.debug(f"Loaded configuration from {source.path}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load configuration from {source.path}: {e}")
                    raise ConfigurationError(
                        f"Failed to load configuration from {source.path}",
                        config_file=str(source.path),
                        details={'error': str(e)}
                    )
            
            # Apply environment variable overrides
            new_config = self._apply_environment_overrides(new_config)
            
            # Validate configuration
            if self.validation_enabled:
                self._validate_configuration(new_config)
            
            self.config = new_config
            self.logger.info("Configuration loaded successfully")
            
            return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self.config:
            self.load_configuration()
        
        return self._get_nested_value(self.config, key, default)
    
    def set(self, key: str, value: Any, persist: bool = False):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            persist: Whether to persist to configuration file
        """
        with self._lock:
            self._set_nested_value(self.config, key, value)
            
            if persist:
                # Find highest priority writable source
                for source in self.sources:
                    if source.path.exists() and os.access(source.path, os.W_OK):
                        self._persist_to_source(source, self.config)
                        break
                else:
                    self.logger.warning("No writable configuration source found for persistence")
        
        # Notify change callbacks
        self._notify_change_callbacks(key, value)
    
    def update(self, updates: Dict[str, Any], persist: bool = False):
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update
            persist: Whether to persist changes to configuration file
        """
        for key, value in updates.items():
            self.set(key, value, persist=False)  # Don't persist individually
        
        if persist:
            # Persist all changes at once
            for source in self.sources:
                if source.path.exists() and os.access(source.path, os.W_OK):
                    self._persist_to_source(source, self.config)
                    break
    
    def reload_if_changed(self) -> bool:
        """
        Reload configuration if any source files have changed.
        
        Returns:
            True if configuration was reloaded, False otherwise
        """
        changed_sources = []
        
        for source in self.sources:
            if not source.path.exists():
                continue
            
            current_mtime = source.path.stat().st_mtime
            current_checksum = self._calculate_file_checksum(source.path)
            
            if (current_mtime != source.last_modified or 
                current_checksum != source.checksum):
                changed_sources.append(source)
                source.last_modified = current_mtime
                source.checksum = current_checksum
        
        if changed_sources:
            self.logger.info(f"Configuration sources changed: {[str(s.path) for s in changed_sources]}")
            self.load_configuration(reload=True)
            return True
        
        return False
    
    def start_change_monitoring(self):
        """Start monitoring configuration files for changes."""
        if self.watching:
            return
        
        self.watching = True
        self.watch_thread = threading.Thread(target=self._change_monitor_loop, daemon=True)
        self.watch_thread.start()
        
        self.logger.info("Started configuration change monitoring")
    
    def stop_change_monitoring(self):
        """Stop monitoring configuration files for changes."""
        self.watching = False
        if self.watch_thread:
            self.watch_thread.join(timeout=5)
        
        self.logger.info("Stopped configuration change monitoring")
    
    def add_change_callback(self, callback: Callable[[str, Any], None]):
        """Add callback for configuration changes."""
        self.change_callbacks.append(callback)
    
    def get_configuration_report(self) -> Dict[str, Any]:
        """Generate comprehensive configuration report."""
        return {
            'environment': self.environment.value,
            'sources': [
                {
                    'path': str(source.path),
                    'format': source.format.value,
                    'priority': source.priority,
                    'environment': source.environment.value if source.environment else 'all',
                    'last_modified': time.ctime(source.last_modified) if source.last_modified else None,
                    'exists': source.path.exists(),
                    'readable': os.access(source.path, os.R_OK) if source.path.exists() else False
                }
                for source in self.sources
            ],
            'schemas_registered': list(self.schemas.keys()),
            'validation_enabled': self.validation_enabled,
            'change_monitoring': self.watching,
            'total_config_keys': len(self._flatten_dict(self.config)),
            'secure_keys_count': len(self.secure_keys)
        }
    
    def export_configuration(self, output_path: Union[str, Path], 
                           format: ConfigurationFormat = ConfigurationFormat.YAML,
                           include_secure: bool = False) -> Path:
        """
        Export current configuration to file.
        
        Args:
            output_path: Output file path
            format: Output format
            include_secure: Whether to include secure keys (not recommended)
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        
        # Filter out secure keys if requested
        config_to_export = self.config.copy()
        if not include_secure:
            config_to_export = self._filter_secure_keys(config_to_export)
        
        # Write in specified format
        if format == ConfigurationFormat.YAML:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_export, f, default_flow_style=False, indent=2)
        elif format == ConfigurationFormat.JSON:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_export, f, indent=2)
        else:
            raise ConfigurationError(f"Export format {format.value} not supported")
        
        self.logger.info(f"Configuration exported to {output_path}")
        return output_path
    
    def _load_source(self, source: ConfigurationSource) -> Dict[str, Any]:
        """Load configuration from a single source."""
        if source.format == ConfigurationFormat.YAML:
            with open(source.path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        elif source.format == ConfigurationFormat.JSON:
            with open(source.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif source.format == ConfigurationFormat.ENV:
            return self._load_env_file(source.path)
        else:
            raise ConfigurationError(f"Unsupported configuration format: {source.format.value}")
    
    def _load_env_file(self, env_path: Path) -> Dict[str, Any]:
        """Load configuration from .env file."""
        config = {}
        
        with open(env_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                
                # Convert value to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
                
                config[key] = value
        
        return config
    
    def _merge_configurations(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configurations(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Look for environment variables that match configuration keys
        env_overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith('CONFIG_'):
                config_key = key[7:].lower().replace('_', '.')  # Remove CONFIG_ prefix
                env_overrides[config_key] = value
        
        # Apply overrides
        for key, value in env_overrides.items():
            self._set_nested_value(config, key, value)
        
        return config
    
    def _validate_configuration(self, config: Dict[str, Any]):
        """Validate configuration against registered schemas."""
        for config_key, schema in self.schemas.items():
            if config_key in config:
                section_config = config[config_key]
                self._validate_section(config_key, section_config, schema)
    
    def _validate_section(self, section_name: str, section_config: Dict[str, Any], 
                         schema: ConfigurationSchema):
        """Validate a configuration section against its schema."""
        # Check required keys
        missing_keys = []
        for key in schema.required_keys:
            if key not in section_config:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValidationError(
                f"Missing required configuration keys in {section_name}: {missing_keys}",
                validation_type="required_keys",
                failed_value=missing_keys
            )
        
        # Check key types and validators
        for key, value in section_config.items():
            if key not in schema.required_keys and key not in schema.optional_keys:
                self.logger.warning(f"Unknown configuration key: {section_name}.{key}")
                continue
            
            # Type validation
            if key in schema.key_types:
                expected_type = schema.key_types[key]
                if not isinstance(value, expected_type):
                    raise ValidationError(
                        f"Invalid type for {section_name}.{key}: expected {expected_type.__name__}, got {type(value).__name__}",
                        validation_type="type_check",
                        failed_value=value
                    )
            
            # Custom validation
            if key in schema.key_validators:
                validator = schema.key_validators[key]
                try:
                    if not validator(value):
                        raise ValidationError(
                            f"Validation failed for {section_name}.{key}",
                            validation_type="custom_validator",
                            failed_value=value
                        )
                except Exception as e:
                    raise ValidationError(
                        f"Validator error for {section_name}.{key}: {str(e)}",
                        validation_type="validator_error",
                        failed_value=value
                    )
        
        # Validate nested schemas
        for key, nested_schema in schema.nested_schemas.items():
            if key in section_config and isinstance(section_config[key], dict):
                self._validate_section(f"{section_name}.{key}", section_config[key], nested_schema)
    
    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any) -> Any:
        """Get nested configuration value using dot notation."""
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = key.split('.')
        target = config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
    
    def _persist_to_source(self, source: ConfigurationSource, config: Dict[str, Any]):
        """Persist configuration to a source file."""
        if source.format == ConfigurationFormat.YAML:
            with open(source.path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif source.format == ConfigurationFormat.JSON:
            with open(source.path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        
        # Update source metadata
        source.last_modified = source.path.stat().st_mtime
        source.checksum = self._calculate_file_checksum(source.path)
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum for change detection."""
        if not file_path.exists():
            return ""
        
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    
    def _change_monitor_loop(self):
        """Main loop for monitoring configuration changes."""
        while self.watching:
            try:
                if self.reload_if_changed():
                    self.logger.info("Configuration reloaded due to file changes")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in change monitor loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _notify_change_callbacks(self, key: str, value: Any):
        """Notify all registered change callbacks."""
        for callback in self.change_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                self.logger.error(f"Error in change callback: {e}")
    
    def _filter_secure_keys(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out secure keys from configuration."""
        filtered = {}
        
        for key, value in config.items():
            if any(secure_key in key.lower() for secure_key in self.secure_keys):
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = self._filter_secure_keys(value)
            else:
                filtered[key] = value
        
        return filtered
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)


# Utility functions
def load_config(config_dir: str = "config", environment: Optional[str] = None) -> Dict[str, Any]:
    """Quick configuration loading utility."""
    manager = ConfigurationManager(config_dir)
    
    # Add common configuration files
    config_files = [
        'app.yaml',
        'logging.yaml', 
        'processing.yaml',
        f'{environment or manager.environment.value}.yaml'
    ]
    
    for config_file in config_files:
        config_path = manager.base_config_dir / config_file
        if config_path.exists():
            manager.add_configuration_source(config_path)
    
    return manager.load_configuration()


def create_default_config(output_dir: str = "config") -> Path:
    """Create default configuration files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Default app configuration
    app_config = {
        'app': {
            'name': 'ASR Post-Processing Workflow',
            'version': '5.3.0',
            'description': 'Advanced ASR Post-Processing for Yoga Vedanta Lectures',
            'debug_mode': False
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_path': 'logs/application.log',
            'max_file_size': 10485760,  # 10MB
            'backup_count': 5
        },
        'data_sources': {
            'raw_srts_path': 'data/raw_srts',
            'processed_srts_path': 'data/processed_srts',
            'lexicons_path': 'data/lexicons'
        },
        'processing': {
            'enable_sanskrit_processing': True,
            'enable_ner': True,
            'enable_mcp_processing': True,
            'performance_monitoring': True,
            'batch_size': 100,
            'max_processing_time_ms': 5000.0
        }
    }
    
    config_file = output_dir / 'app.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(app_config, f, default_flow_style=False, indent=2)
    
    return config_file


# Test function
def test_config_manager():
    """Test configuration manager functionality."""
    import tempfile
    import shutil
    
    print("Testing Configuration Manager...")
    
    # Create temporary directory for test configs
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test configuration
        config_file = create_default_config(temp_dir)
        print(f"Created default config: {config_file}")
        
        # Initialize manager
        manager = ConfigurationManager(temp_dir)
        manager.add_configuration_source(config_file)
        
        # Load and test configuration
        config = manager.load_configuration()
        print(f"Loaded configuration with {len(manager._flatten_dict(config))} keys")
        
        # Test getting values
        app_name = manager.get('app.name')
        print(f"App name: {app_name}")
        
        # Test setting values
        manager.set('app.debug_mode', True)
        debug_mode = manager.get('app.debug_mode')
        print(f"Debug mode after update: {debug_mode}")
        
        # Test validation
        print("Testing validation...")
        try:
            manager.set('logging.level', 'INVALID')
            manager._validate_configuration(manager.config)
        except ValidationError as e:
            print(f"Validation caught invalid value: {e.message}")
        
        # Test configuration report
        report = manager.get_configuration_report()
        print(f"Configuration report: {report['total_config_keys']} keys, {len(report['sources'])} sources")
        
        print("Configuration manager test completed successfully!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_config_manager()