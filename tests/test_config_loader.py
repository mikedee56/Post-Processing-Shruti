"""
Unit tests for ConfigLoader class.

Tests the configuration loading, validation, and error handling functionality.
"""

import os
import pytest
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.config_loader import ConfigLoader, ProcessingConfig


class TestConfigLoader:
    """Test cases for ConfigLoader class."""
    
    def test_initialization_success(self):
        """Test that ConfigLoader initializes successfully without errors."""
        # This should not raise AttributeError
        config_loader = ConfigLoader()
        
        # The exists() method should work
        assert config_loader.exists() is True
        assert isinstance(config_loader.get_config(), ProcessingConfig)
    
    def test_exists_method(self):
        """Test the exists() method functionality."""
        config_loader = ConfigLoader()
        
        # Should return True when properly initialized
        assert config_loader.exists() is True
        
        # Should return False if config is None
        config_loader.config = None
        assert config_loader.exists() is False
        
        # Should return False if config is not ProcessingConfig instance
        config_loader.config = {"not": "a ProcessingConfig"}
        assert config_loader.exists() is False
    
    def test_default_configuration_values(self):
        """Test that default configuration values are set properly."""
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        # Test default values
        assert config.fuzzy_threshold == 80
        assert config.confidence_threshold == 0.6
        assert config.batch_size == 10
        assert config.max_concurrent_files == 3
        
        # Test lexicon paths
        assert 'corrections' in config.lexicon_paths
        assert 'proper_nouns' in config.lexicon_paths
        assert 'phrases' in config.lexicon_paths
        assert 'verses' in config.lexicon_paths
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config_loader = ConfigLoader()
        
        # Default configuration should be valid
        assert config_loader.validate_config() is True
        
        # Test invalid fuzzy threshold
        config_loader.config.fuzzy_threshold = 150
        assert config_loader.validate_config() is False
        
        # Reset to valid value
        config_loader.config.fuzzy_threshold = 80
        assert config_loader.validate_config() is True
        
        # Test invalid confidence threshold
        config_loader.config.confidence_threshold = 1.5
        assert config_loader.validate_config() is False
    
    def test_yaml_config_file_loading(self):
        """Test loading configuration from YAML file."""
        test_config = {
            'fuzzy_threshold': 90,
            'confidence_threshold': 0.8,
            'batch_size': 20,
            'logging_level': 'DEBUG'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_file = Path(f.name)
        
        try:
            config_loader = ConfigLoader(config_file)
            config = config_loader.get_config()
            
            assert config.fuzzy_threshold == 90
            assert config.confidence_threshold == 0.8
            assert config.batch_size == 20
            assert config.logging_level == 'DEBUG'
            
        finally:
            config_file.unlink()
    
    def test_json_config_file_loading(self):
        """Test loading configuration from JSON file."""
        test_config = {
            'fuzzy_threshold': 85,
            'confidence_threshold': 0.7,
            'batch_size': 15
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_file = Path(f.name)
        
        try:
            config_loader = ConfigLoader(config_file)
            config = config_loader.get_config()
            
            assert config.fuzzy_threshold == 85
            assert config.confidence_threshold == 0.7
            assert config.batch_size == 15
            
        finally:
            config_file.unlink()
    
    def test_missing_config_file_graceful_handling(self):
        """Test graceful handling when config file is missing."""
        non_existent_file = Path('/non/existent/config.yaml')
        
        # Should not raise an exception
        config_loader = ConfigLoader(non_existent_file)
        
        # Should still be valid with defaults
        assert config_loader.exists() is True
        assert isinstance(config_loader.get_config(), ProcessingConfig)
    
    def test_environment_variable_overrides(self):
        """Test that environment variables override configuration values."""
        with patch.dict(os.environ, {
            'ASR_FUZZY_THRESHOLD': '95',
            'ASR_CONFIDENCE_THRESHOLD': '0.9',
            'ASR_BATCH_SIZE': '25'
        }):
            config_loader = ConfigLoader()
            config = config_loader.get_config()
            
            assert config.fuzzy_threshold == 95
            assert config.confidence_threshold == 0.9
            assert config.batch_size == 25
    
    def test_corrupted_config_file_handling(self):
        """Test handling of corrupted/invalid config files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            config_file = Path(f.name)
        
        try:
            # Should handle the error gracefully and continue with defaults
            with patch('config.config_loader.logging') as mock_logging:
                config_loader = ConfigLoader(config_file)
                # Should still create a valid config object
                assert config_loader.exists() is True
                
        finally:
            config_file.unlink()
    
    def test_lexicon_path_validation(self):
        """Test lexicon file path validation and fallback."""
        config_loader = ConfigLoader()
        
        # Call the validation method directly
        config_loader._validate_and_setup_lexicon_paths()
        
        # Should not raise exceptions
        assert config_loader.exists() is True
        config = config_loader.get_config()
        assert 'corrections' in config.lexicon_paths
    
    def test_get_lexicon_config(self):
        """Test getting lexicon configuration."""
        config_loader = ConfigLoader()
        lexicon_config = config_loader.get_lexicon_config()
        
        assert isinstance(lexicon_config, dict)
        assert 'corrections' in lexicon_config
        assert 'proper_nouns' in lexicon_config
        assert 'phrases' in lexicon_config
        assert 'verses' in lexicon_config
    
    def test_get_storage_config(self):
        """Test getting storage configuration."""
        config_loader = ConfigLoader()
        storage_config = config_loader.get_storage_config()
        
        assert isinstance(storage_config, dict)
        assert 'input_dir' in storage_config
        assert 'output_dir' in storage_config
        assert 'backup_enabled' in storage_config
    
    def test_get_processing_config(self):
        """Test getting processing configuration."""
        config_loader = ConfigLoader()
        processing_config = config_loader.get_processing_config()
        
        assert isinstance(processing_config, dict)
        assert 'fuzzy_threshold' in processing_config
        assert 'confidence_threshold' in processing_config
        assert 'batch_size' in processing_config
        assert 'max_concurrent_files' in processing_config
    
    def test_configuration_loading_with_exception(self):
        """Test configuration loading handles exceptions gracefully."""
        with patch('config.config_loader.Path.exists', side_effect=Exception("File system error")):
            # Should not raise exception, should fallback to defaults
            config_loader = ConfigLoader()
            assert config_loader.exists() is True
            assert isinstance(config_loader.get_config(), ProcessingConfig)
    
    def test_nested_config_setting(self):
        """Test setting nested configuration values."""
        config_loader = ConfigLoader()
        
        # Test setting a nested value
        config_loader._set_nested_config('storage_config.input_dir', '/new/input/dir')
        config = config_loader.get_config()
        
        assert config.storage_config['input_dir'] == '/new/input/dir'
    
    def test_config_to_dict_conversion(self):
        """Test converting configuration to dictionary."""
        config_loader = ConfigLoader()
        config_dict = config_loader._config_to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'fuzzy_threshold' in config_dict
        assert 'lexicon_paths' in config_dict
        assert 'storage_config' in config_dict


class TestConfigLoaderIntegration:
    """Integration tests for ConfigLoader with other components."""
    
    def test_component_configuration_access(self):
        """Test that components can access their configurations through ConfigLoader."""
        config_loader = ConfigLoader()
        
        # Simulate component accessing configuration
        config = config_loader.get_config()
        
        # Test Sanskrit processor would need these settings
        assert hasattr(config, 'fuzzy_threshold')
        assert hasattr(config, 'confidence_threshold')
        assert hasattr(config, 'lexicon_paths')
        
        # Test IAST transliterator would need these settings
        assert hasattr(config, 'iast_rules')
        
        # Test preprocessing would need these settings
        assert hasattr(config, 'exclusion_patterns')
    
    def test_configuration_persistence(self):
        """Test saving and loading configuration."""
        config_loader = ConfigLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            # Save configuration
            config_loader.save_config(config_path, 'yaml')
            assert config_path.exists()
            
            # Load the saved configuration
            new_config_loader = ConfigLoader(config_path)
            new_config = new_config_loader.get_config()
            
            # Should have the same values
            original_config = config_loader.get_config()
            assert new_config.fuzzy_threshold == original_config.fuzzy_threshold
            assert new_config.confidence_threshold == original_config.confidence_threshold


# Test scenario from the story
def test_story_scenario():
    """Test the specific scenario mentioned in the story."""
    # Test 1: Happy path configuration loading
    config = ConfigLoader()
    assert config.exists()  # Should not throw AttributeError
    
    # Test 2: Component configuration access
    processing_config = config.get_processing_config()
    assert processing_config is not None
    assert 'fuzzy_threshold' in processing_config


if __name__ == '__main__':
    pytest.main([__file__, '-v'])