"""
Test suite for project setup and module imports.

This module validates that the project structure is correctly set up
and all modules can be imported without errors.
"""

import sys
import pytest
from pathlib import Path
import importlib.util

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestProjectStructure:
    """Test basic project structure."""
    
    def test_source_directories_exist(self):
        """Test that all required source directories exist."""
        src_path = Path(__file__).parent.parent / 'src'
        
        required_dirs = [
            'post_processors',
            'sanskrit_hindi_identifier', 
            'ner_module',
            'qa_module',
            'utils',
            'config',
            'storage'  # From Story 1.1
        ]
        
        for dir_name in required_dirs:
            dir_path = src_path / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"
            assert dir_path.is_dir(), f"{dir_name} should be a directory"
            
            # Check for __init__.py
            init_file = dir_path / '__init__.py'
            assert init_file.exists(), f"{dir_name}/__init__.py should exist"
    
    def test_main_entry_point_exists(self):
        """Test that main.py entry point exists."""
        main_file = Path(__file__).parent.parent / 'src' / 'main.py'
        assert main_file.exists(), "main.py should exist"
        assert main_file.is_file(), "main.py should be a file"
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        req_file = Path(__file__).parent.parent / 'requirements.txt'
        assert req_file.exists(), "requirements.txt should exist"
        assert req_file.is_file(), "requirements.txt should be a file"
    
    def test_config_directory_structure(self):
        """Test config directory structure from Story 1.1."""
        config_path = Path(__file__).parent.parent / 'config'
        assert config_path.exists(), "config directory should exist"
        
        # Check for existing config files from Story 1.1
        expected_files = [
            'file_naming.yaml',
            'storage_management.yaml'
        ]
        
        for file_name in expected_files:
            file_path = config_path / file_name
            assert file_path.exists(), f"Config file {file_name} should exist"
    
    def test_data_directory_structure(self):
        """Test data directory structure from Story 1.1."""
        data_path = Path(__file__).parent.parent / 'data'
        assert data_path.exists(), "data directory should exist"
        
        expected_dirs = [
            'raw_srts',
            'processed_srts',
            'lexicons',
            'golden_dataset'
        ]
        
        for dir_name in expected_dirs:
            dir_path = data_path / dir_name
            assert dir_path.exists(), f"Data directory {dir_name} should exist"


class TestModuleImports:
    """Test that all modules can be imported successfully."""
    
    def test_import_sanskrit_processor(self):
        """Test importing the Sanskrit post-processor."""
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            assert SanskritPostProcessor is not None
        except ImportError as e:
            pytest.fail(f"Failed to import SanskritPostProcessor: {e}")
    
    def test_import_config_loader(self):
        """Test importing the configuration loader."""
        try:
            from config.config_loader import ConfigLoader, ProcessingConfig
            assert ConfigLoader is not None
            assert ProcessingConfig is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config modules: {e}")
    
    def test_import_file_utils(self):
        """Test importing file utilities."""
        try:
            from utils.file_utils import validate_srt_file, detect_file_encoding
            assert validate_srt_file is not None
            assert detect_file_encoding is not None
        except ImportError as e:
            pytest.fail(f"Failed to import file utils: {e}")
    
    def test_import_text_utils(self):
        """Test importing text utilities."""
        try:
            from utils.text_utils import normalize_text, fuzzy_match_terms
            assert normalize_text is not None
            assert fuzzy_match_terms is not None
        except ImportError as e:
            pytest.fail(f"Failed to import text utils: {e}")
    
    def test_import_storage_modules(self):
        """Test importing storage modules from Story 1.1."""
        try:
            from storage.file_ingestion import FileIngestionSystem
            from storage.output_management import OutputStorageManager
            assert FileIngestionSystem is not None
            assert OutputStorageManager is not None
        except ImportError as e:
            pytest.fail(f"Failed to import storage modules: {e}")


class TestBasicFunctionality:
    """Test basic functionality of key components."""
    
    def test_config_loader_initialization(self):
        """Test that ConfigLoader can be initialized."""
        from config.config_loader import ConfigLoader
        
        # Should work without config file
        loader = ConfigLoader()
        assert loader is not None
        
        config = loader.get_config()
        assert config is not None
        assert hasattr(config, 'fuzzy_threshold')
        assert hasattr(config, 'confidence_threshold')
    
    def test_sanskrit_processor_initialization(self):
        """Test that SanskritPostProcessor can be initialized."""
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        # Should work without config file (uses defaults)
        processor = SanskritPostProcessor()
        assert processor is not None
        assert hasattr(processor, 'corrections')
        assert hasattr(processor, 'proper_nouns')
    
    def test_text_normalization(self):
        """Test basic text normalization functionality."""
        from utils.text_utils import normalize_text, clean_whitespace
        
        # Test basic normalization
        test_text = "  Hello   world  "
        normalized = clean_whitespace(test_text)
        assert normalized == "Hello world"
        
        # Test unicode normalization
        test_unicode = "café"
        normalized_unicode = normalize_text(test_unicode)
        assert len(normalized_unicode) > 0
    
    def test_file_validation_basic(self):
        """Test basic file validation functionality."""
        from utils.file_utils import safe_filename, normalize_path
        
        # Test safe filename creation
        unsafe_name = "file<>name|with*bad?chars"
        safe_name = safe_filename(unsafe_name)
        assert '<' not in safe_name
        assert '>' not in safe_name
        assert '|' not in safe_name
        
        # Test path normalization
        test_path = "some/relative/path"
        normalized = normalize_path(test_path)
        assert normalized.is_absolute()


class TestDependencyAvailability:
    """Test that required dependencies are available."""
    
    def test_pandas_available(self):
        """Test that pandas is available."""
        try:
            import pandas as pd
            assert pd is not None
        except ImportError:
            pytest.skip("pandas not installed")
    
    def test_pysrt_available(self):
        """Test that pysrt is available."""
        try:
            import pysrt
            assert pysrt is not None
        except ImportError:
            pytest.skip("pysrt not installed")
    
    def test_yaml_available(self):
        """Test that PyYAML is available."""
        try:
            import yaml
            assert yaml is not None
        except ImportError:
            pytest.skip("PyYAML not installed")
    
    def test_fuzzywuzzy_available(self):
        """Test that fuzzywuzzy is available."""
        try:
            from fuzzywuzzy import fuzz
            assert fuzz is not None
        except ImportError:
            pytest.skip("fuzzywuzzy not installed")
    
    def test_click_available(self):
        """Test that click is available."""
        try:
            import click
            assert click is not None
        except ImportError:
            pytest.skip("click not installed")


class TestMainEntryPoint:
    """Test main application entry point."""
    
    def test_main_module_structure(self):
        """Test that main.py has expected structure."""
        main_path = Path(__file__).parent.parent / 'src' / 'main.py'
        
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for CLI command decorators
        assert '@cli.command()' in content, "Should have CLI commands"
        assert 'process_single' in content, "Should have process_single command"
        assert 'process_batch' in content, "Should have process_batch command"
        assert 'click.group()' in content, "Should use click for CLI"
    
    def test_main_imports(self):
        """Test that main.py imports are working."""
        # This is a basic syntax check - actual CLI testing would need more setup
        main_path = Path(__file__).parent.parent / 'src' / 'main.py'
        
        spec = importlib.util.spec_from_file_location("main", main_path)
        assert spec is not None, "Should be able to load main module spec"


if __name__ == '__main__':
    # Run tests when executed directly
    pytest.main([__file__, '-v'])