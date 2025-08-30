"""
Configuration management module for ASR Post-Processing Workflow.

This module provides centralized configuration loading and management
for the post-processing pipeline components.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, field


@dataclass
class ProcessingConfig:
    """Configuration for post-processing pipeline."""
    
    # File paths
    lexicon_paths: Dict[str, str] = field(default_factory=lambda: {
        'corrections': 'data/lexicons/corrections.yaml',
        'proper_nouns': 'data/lexicons/proper_nouns.yaml',
        'phrases': 'data/lexicons/phrases.yaml',
        'verses': 'data/lexicons/verses.yaml'
    })
    
    # Processing thresholds
    fuzzy_threshold: int = 80
    confidence_threshold: float = 0.6
    
    # Batch processing
    batch_size: int = 10
    max_concurrent_files: int = 3
    
    # Quality assurance
    flag_low_confidence: bool = True
    flag_high_oov: bool = True
    oov_threshold: float = 0.3
    
    # IAST transliteration rules
    iast_rules: Dict[str, Any] = field(default_factory=lambda: {
        'apply_strict_iast': True,
        'preserve_diacritics': True,
        'normalize_variants': True
    })
    
    # Exclusion patterns
    exclusion_patterns: list = field(default_factory=lambda: [
        r'\b[A-Z]{2,}\b',  # Acronyms
        r'\b\d+\b',        # Numbers
        r'\b[a-z]+\.[a-z]+\b'  # Email-like patterns
    ])
    
    # Logging configuration
    logging_level: str = 'INFO'
    log_file: Optional[str] = 'logs/processing.log'
    
    # Storage settings
    storage_config: Dict[str, Any] = field(default_factory=lambda: {
        'input_dir': 'data/raw_srts',
        'output_dir': 'data/processed_srts',
        'backup_enabled': True,
        'backup_dir': 'data/backups',
        'preserve_structure': True
    })
    
    # Story 3.5: Semantic feature configuration with gradual rollout support
    semantic_features: Dict[str, Any] = field(default_factory=lambda: {
        # Master feature flag for all semantic capabilities
        'enable_semantic_features': False,
        
        # Gradual rollout flags - allows per-feature enablement
        'feature_flags': {
            'semantic_analysis': False,          # Story 3.1 - Core semantic processing
            'domain_classification': False,      # Story 3.1 - Domain-specific processing  
            'academic_qa_framework': False,      # Story 3.2 - Quality gates
            'expert_review_queue': False,        # Story 3.2.1 - Expert review routing
            'term_relationship_mapping': False,  # Story 3.1 - Relationship graph
            'contextual_validation': False,      # Story 3.1 - Translation validation
            'performance_monitoring': True,      # Story 3.4 - Always enabled for metrics
        },
        
        # Rollout percentages (0-100) for A/B testing
        'rollout_percentages': {
            'semantic_analysis': 0,
            'domain_classification': 0, 
            'academic_qa_framework': 0,
            'expert_review_queue': 0,
            'term_relationship_mapping': 0,
            'contextual_validation': 0
        },
        
        # Performance thresholds
        'performance_limits': {
            'max_semantic_processing_time_ms': 100,  # Per segment
            'max_cache_miss_ratio': 0.05,           # 95% cache hit requirement
            'max_memory_usage_mb': 512,             # Memory limit
            'circuit_breaker_threshold': 5,         # Failures before fallback
        },
        
        # Infrastructure settings
        'infrastructure': {
            'redis_enabled': True,
            'vector_database_enabled': False,  # Gradual rollout
            'batch_processing_enabled': True,
            'graceful_degradation_enabled': True,
        },
        
        # Backward compatibility settings
        'compatibility': {
            'preserve_legacy_api': True,         # Keep existing API contracts
            'legacy_fallback_enabled': True,    # Fallback to pre-semantic processing
            'maintain_output_format': True,     # Preserve existing output structure
            'performance_regression_threshold': 0.05,  # Max 5% performance impact
        }
    })
    
    # Academic validation settings (Story 3.2)
    academic_validation: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,  # Gradual rollout
        'quality_thresholds': {
            'minimum_quality_score': 0.7,
            'minimum_compliance_score': 0.8, 
            'minimum_iast_compliance': 0.8,
        },
        'validation_timeout_ms': 50,  # Story 3.2 requirement
        'expert_review_routing': {
            'complexity_threshold': 0.8,
            'confidence_threshold': 0.6,
            'automatic_routing_enabled': False,  # Manual control initially
        }
    })


class ConfigLoader:
    """
    Configuration loader and manager for the ASR post-processing system.
    
    Handles loading configuration from multiple sources with precedence:
    1. Command-line arguments
    2. Environment variables
    3. Configuration files (YAML/JSON)
    4. Default values
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.config = ProcessingConfig()
        
        # Load configuration from various sources
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from all available sources."""
        
        # 1. Load from default configuration file if no specific file provided
        if not self.config_file:
            default_config_paths = [
                Path('config/processing.yaml'),
                Path('config/processing.json'),
                Path('processing.yaml'),
                Path('processing.json')
            ]
            
            for path in default_config_paths:
                if path.exists():
                    self.config_file = path
                    break
        
        # 2. Load from configuration file
        if self.config_file and self.config_file.exists():
            self._load_from_file(self.config_file)
        
        # 3. Override with environment variables
        self._load_from_environment()
        
        self.logger.info(f"Configuration loaded from: {self.config_file or 'defaults'}")
    
    def _load_from_file(self, config_path: Path):
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    self.logger.warning(f"Unsupported config file format: {config_path}")
                    return
            
            # Update configuration with file values
            self._update_config_from_dict(file_config)
            
            self.logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    def _load_from_environment(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'ASR_FUZZY_THRESHOLD': ('fuzzy_threshold', int),
            'ASR_CONFIDENCE_THRESHOLD': ('confidence_threshold', float),
            'ASR_BATCH_SIZE': ('batch_size', int),
            'ASR_MAX_CONCURRENT': ('max_concurrent_files', int),
            'ASR_LOG_LEVEL': ('logging_level', str),
            'ASR_INPUT_DIR': ('storage_config.input_dir', str),
            'ASR_OUTPUT_DIR': ('storage_config.output_dir', str),
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self._set_nested_config(config_key, converted_value)
                    self.logger.debug(f"Set {config_key} = {converted_value} from {env_var}")
                except ValueError as e:
                    self.logger.warning(f"Invalid value for {env_var}: {value} ({e})")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration object from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                if isinstance(getattr(self.config, key), dict) and isinstance(value, dict):
                    # Merge dictionaries
                    current_dict = getattr(self.config, key)
                    current_dict.update(value)
                else:
                    setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def _set_nested_config(self, key_path: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                self.logger.warning(f"Configuration path not found: {key_path}")
                return
        
        final_key = keys[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        elif isinstance(current, dict):
            current[final_key] = value
        else:
            self.logger.warning(f"Cannot set configuration: {key_path}")
    
    def get_config(self) -> ProcessingConfig:
        """Get the loaded configuration object."""
        return self.config
    
    def get_lexicon_config(self) -> Dict[str, str]:
        """Get lexicon file paths configuration."""
        return self.config.lexicon_paths
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return self.config.storage_config
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing-specific configuration."""
        return {
            'fuzzy_threshold': self.config.fuzzy_threshold,
            'confidence_threshold': self.config.confidence_threshold,
            'batch_size': self.config.batch_size,
            'max_concurrent_files': self.config.max_concurrent_files,
            'flag_low_confidence': self.config.flag_low_confidence,
            'flag_high_oov': self.config.flag_high_oov,
            'oov_threshold': self.config.oov_threshold
        }
    
    def validate_config(self) -> bool:
        """Validate the loaded configuration."""
        validation_errors = []
        
        # Validate thresholds
        if not 0 <= self.config.fuzzy_threshold <= 100:
            validation_errors.append("fuzzy_threshold must be between 0 and 100")
        
        if not 0.0 <= self.config.confidence_threshold <= 1.0:
            validation_errors.append("confidence_threshold must be between 0.0 and 1.0")
        
        if self.config.batch_size <= 0:
            validation_errors.append("batch_size must be positive")
        
        if self.config.max_concurrent_files <= 0:
            validation_errors.append("max_concurrent_files must be positive")
        
        # Validate paths
        for lexicon_type, path in self.config.lexicon_paths.items():
            lexicon_path = Path(path)
            if not lexicon_path.parent.exists():
                validation_errors.append(f"Lexicon directory does not exist: {lexicon_path.parent}")
        
        # Log validation results
        if validation_errors:
            for error in validation_errors:
                self.logger.error(f"Configuration validation error: {error}")
            return False
        else:
            self.logger.info("Configuration validation passed")
            return True
    
    def save_config(self, output_path: Path, format: str = 'yaml'):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        try:
            config_dict = self._config_to_dict()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration to {output_path}: {e}")
            raise
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return {
            'lexicon_paths': self.config.lexicon_paths,
            'fuzzy_threshold': self.config.fuzzy_threshold,
            'confidence_threshold': self.config.confidence_threshold,
            'batch_size': self.config.batch_size,
            'max_concurrent_files': self.config.max_concurrent_files,
            'flag_low_confidence': self.config.flag_low_confidence,
            'flag_high_oov': self.config.flag_high_oov,
            'oov_threshold': self.config.oov_threshold,
            'iast_rules': self.config.iast_rules,
            'exclusion_patterns': self.config.exclusion_patterns,
            'logging_level': self.config.logging_level,
            'log_file': self.config.log_file,
            'storage_config': self.config.storage_config
        }
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        print("=" * 50)
        print("ASR Post-Processing Configuration Summary")
        print("=" * 50)
        
        print(f"🔧 Processing Settings:")
        print(f"  Fuzzy Threshold: {self.config.fuzzy_threshold}")
        print(f"  Confidence Threshold: {self.config.confidence_threshold}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Max Concurrent Files: {self.config.max_concurrent_files}")
        
        print(f"\n📚 Lexicon Files:")
        for lexicon_type, path in self.config.lexicon_paths.items():
            exists = "✅" if Path(path).exists() else "❌"
            print(f"  {lexicon_type.title()}: {path} {exists}")
        
        print(f"\n💾 Storage Settings:")
        print(f"  Input Directory: {self.config.storage_config['input_dir']}")
        print(f"  Output Directory: {self.config.storage_config['output_dir']}")
        print(f"  Backup Enabled: {self.config.storage_config['backup_enabled']}")
        
        print(f"\n📝 Logging:")
        print(f"  Level: {self.config.logging_level}")
        print(f"  Log File: {self.config.log_file or 'None'}")
        
        print("=" * 50)