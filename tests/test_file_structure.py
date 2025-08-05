"""
Test Suite for File Structure and Storage Management
Story 1.1: File Naming Conventions & Storage Management

This module contains unit and integration tests for file naming validation,
directory structure creation, and file system operations.
"""

import unittest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open

# Import modules under test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.file_ingestion import FileIngestionSystem, FileInfo
from src.storage.output_management import OutputStorageManager, ProcessingMetadata


class TestFileNamingValidation(unittest.TestCase):
    """Unit tests for file naming convention validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock config
        self.config = {
            'file_naming': {
                'srt_files': {
                    'pattern': '{lecture_id}_{date}_{speaker}.srt'
                }
            },
            'directories': {
                'input': 'data/raw_srts',
                'output': 'data/processed_srts'
            },
            'validation': {
                'srt_files': {
                    'max_size_mb': 50,
                    'required_extensions': ['.srt']
                },
                'naming_constraints': {
                    'max_filename_length': 255,
                    'forbidden_chars': ['<', '>', ':', '"', '|', '?', '*']
                }
            }
        }
        
        # Create config file
        self.config_path = self.temp_path / "test_config.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_valid_filename_parsing(self):
        """Test parsing of valid filenames."""
        with patch('src.storage.file_ingestion.FileIngestionSystem._load_config', return_value=self.config):
            ingestion = FileIngestionSystem()
            
            # Test valid filename
            lecture_id, date, speaker, is_valid, errors = ingestion._parse_filename(
                "VED001_20241201_SwamiBrahmananda.srt"
            )
            
            self.assertTrue(is_valid)
            self.assertEqual(lecture_id, "VED001")
            self.assertEqual(date, "20241201")
            self.assertEqual(speaker, "SwamiBrahmananda")
            self.assertEqual(len(errors), 0)
    
    def test_invalid_filename_formats(self):
        """Test parsing of invalid filenames."""
        with patch('src.storage.file_ingestion.FileIngestionSystem._load_config', return_value=self.config):
            ingestion = FileIngestionSystem()
            
            # Test cases for invalid filenames
            invalid_cases = [
                "invalid_format.srt",  # Wrong pattern
                "VED1_20241201_Speaker.srt",  # Wrong lecture_id format
                "VED001_2024-12-01_Speaker.srt",  # Wrong date format
                "VED001_20241201_speaker_name.srt",  # Wrong speaker format
                "VED001_20241301_Speaker.srt",  # Invalid date
            ]
            
            for filename in invalid_cases:
                _, _, _, is_valid, errors = ingestion._parse_filename(filename)
                self.assertFalse(is_valid, f"Expected {filename} to be invalid")
                self.assertGreater(len(errors), 0, f"Expected errors for {filename}")
    
    def test_processed_filename_generation(self):
        """Test generation of processed filenames."""
        with patch('src.storage.output_management.OutputStorageManager._load_config', return_value=self.config):
            output_manager = OutputStorageManager()
            
            # Test processed filename generation
            original = "VED001_20241201_SwamiBrahmananda.srt"
            processed = output_manager.generate_processed_filename(original, version=1)
            expected = "VED001_20241201_SwamiBrahmananda_processed_v1.srt"
            
            self.assertEqual(processed, expected)
            
            # Test with different version
            processed_v2 = output_manager.generate_processed_filename(original, version=2)
            expected_v2 = "VED001_20241201_SwamiBrahmananda_processed_v2.srt"
            
            self.assertEqual(processed_v2, expected_v2)


class TestDirectoryStructureOperations(unittest.TestCase):
    """Integration tests for directory structure creation and management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directory structure
        self.raw_srts = self.temp_path / "data" / "raw_srts"
        self.processed_srts = self.temp_path / "data" / "processed_srts"
        self.logs = self.temp_path / "logs"
        
        self.raw_srts.mkdir(parents=True)
        
        # Create mock config
        self.config = {
            'directories': {
                'input': 'data/raw_srts',
                'output': 'data/processed_srts'
            },
            'validation': {
                'srt_files': {
                    'max_size_mb': 50,
                    'required_extensions': ['.srt']
                },
                'naming_constraints': {
                    'max_filename_length': 255,
                    'forbidden_chars': ['<', '>', ':', '"', '|', '?', '*']
                }
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_directory_mirroring(self):
        """Test directory structure mirroring from input to output."""
        with patch('src.storage.output_management.OutputStorageManager._load_config', return_value=self.config):
            # Change to temp directory for relative path operations
            original_cwd = Path.cwd()
            os.chdir(self.temp_path)
            
            try:
                output_manager = OutputStorageManager()
                
                # Create nested directory structure in input
                nested_input = self.raw_srts / "subfolder" / "deep"
                nested_input.mkdir(parents=True)
                
                # Create test file
                test_file = nested_input / "VED001_20241201_SwamiBrahmananda.srt"
                test_file.write_text("Test SRT content")
                
                # Test mirroring
                mirrored_path = output_manager.mirror_directory_structure(test_file)
                expected_path = self.processed_srts / "subfolder" / "deep" / "VED001_20241201_SwamiBrahmananda.srt"
                
                self.assertEqual(mirrored_path, expected_path)
                self.assertTrue(mirrored_path.parent.exists())
                
            finally:
                os.chdir(original_cwd)
    
    def test_file_discovery(self):
        """Test file discovery in directory structure."""
        with patch('src.storage.file_ingestion.FileIngestionSystem._load_config', return_value=self.config):
            # Change to temp directory for relative path operations
            original_cwd = Path.cwd()
            os.chdir(self.temp_path)
            
            try:
                # Create test files
                test_files = [
                    "VED001_20241201_SwamiBrahmananda.srt",
                    "VED002_20241202_SwamiBrahmavidyananda.srt",
                    "subfolder/VED003_20241203_AcharyaRamesh.srt"
                ]
                
                for file_path in test_files:
                    full_path = self.raw_srts / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text("Test SRT content\n1\n00:00:00,000 --> 00:00:05,000\nTest")
                
                # Test file discovery
                ingestion = FileIngestionSystem()
                discovered_files = ingestion.discover_files()
                
                self.assertEqual(len(discovered_files), 3)
                
                # Check that all files were discovered
                discovered_names = [f.filename for f in discovered_files]
                expected_names = [Path(f).name for f in test_files]
                
                for expected_name in expected_names:
                    self.assertIn(expected_name, discovered_names)
                
            finally:
                os.chdir(original_cwd)


class TestFileSystemOperations(unittest.TestCase):
    """Tests for file system operations and validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.config = {
            'directories': {
                'input': 'data/raw_srts',
                'output': 'data/processed_srts'
            },
            'validation': {
                'srt_files': {
                    'max_size_mb': 1,  # Small limit for testing
                    'required_extensions': ['.srt']
                },
                'naming_constraints': {
                    'max_filename_length': 50,  # Small limit for testing
                    'forbidden_chars': ['<', '>', ':', '"', '|', '?', '*']
                }
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_file_size_validation(self):
        """Test file size validation."""
        with patch('src.storage.file_ingestion.FileIngestionSystem._load_config', return_value=self.config):
            # Create oversized file
            large_file = self.temp_path / "large_file.srt"
            large_content = "x" * (2 * 1024 * 1024)  # 2MB content
            large_file.write_text(large_content)
            
            ingestion = FileIngestionSystem()
            errors = ingestion._validate_file(large_file)
            
            # Should have size error
            size_errors = [e for e in errors if "size" in e.lower()]
            self.assertGreater(len(size_errors), 0)
    
    def test_filename_length_validation(self):
        """Test filename length validation."""
        with patch('src.storage.file_ingestion.FileIngestionSystem._load_config', return_value=self.config):
            # Create file with long name
            long_name = "x" * 60 + ".srt"  # Exceeds 50 char limit
            long_file = self.temp_path / long_name
            long_file.write_text("test content")
            
            ingestion = FileIngestionSystem()
            errors = ingestion._validate_file(long_file)
            
            # Should have length error
            length_errors = [e for e in errors if "long" in e.lower()]
            self.assertGreater(len(length_errors), 0)
    
    def test_forbidden_characters_validation(self):
        """Test forbidden characters validation."""
        with patch('src.storage.file_ingestion.FileIngestionSystem._load_config', return_value=self.config):
            # Create file with forbidden character
            forbidden_file = self.temp_path / "test<file>.srt"
            forbidden_file.write_text("test content")
            
            ingestion = FileIngestionSystem()
            errors = ingestion._validate_file(forbidden_file)
            
            # Should have forbidden character error
            char_errors = [e for e in errors if "forbidden" in e.lower()]
            self.assertGreater(len(char_errors), 0)


class TestProcessingMetadata(unittest.TestCase):
    """Tests for processing metadata and logging."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.config = {
            'directories': {
                'input': 'data/raw_srts',
                'output': 'data/processed_srts'
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_processing_metadata_creation(self):
        """Test creation and storage of processing metadata."""
        with patch('src.storage.output_management.OutputStorageManager._load_config', return_value=self.config):
            # Change to temp directory
            original_cwd = Path.cwd()
            os.chdir(self.temp_path)
            
            try:
                # Create input structure
                input_dir = self.temp_path / "data" / "raw_srts"
                input_dir.mkdir(parents=True)
                
                # Create test file
                source_file = input_dir / "VED001_20241201_SwamiBrahmananda.srt"
                source_file.write_text("Original content")
                
                output_manager = OutputStorageManager()
                processed_content = "Processed content"
                
                # Store processed file
                output_path, metadata = output_manager.store_processed_file(
                    source_file, 
                    processed_content,
                    {'test_metric': 0.95}
                )
                
                # Verify metadata
                self.assertIsInstance(metadata, ProcessingMetadata)
                self.assertTrue(metadata.success)
                self.assertEqual(metadata.original_file, str(source_file))
                self.assertEqual(metadata.processed_file, str(output_path))
                self.assertIn('test_metric', metadata.quality_metrics)
                
                # Verify file was created
                self.assertTrue(output_path.exists())
                self.assertEqual(output_path.read_text(), processed_content)
                
                # Verify log was created
                logs_dir = self.temp_path / "logs"
                self.assertTrue(logs_dir.exists())
                
                log_files = list(logs_dir.glob("processing_log_*.json"))
                self.assertGreater(len(log_files), 0)
                
            finally:
                os.chdir(original_cwd)


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create complete directory structure
        directories = [
            "data/raw_srts",
            "data/processed_srts", 
            "data/lexicons",
            "data/golden_dataset",
            "config",
            "logs"
        ]
        
        for directory in directories:
            (self.temp_path / directory).mkdir(parents=True)
        
        # Create config
        self.config = {
            'file_naming': {
                'srt_files': {
                    'pattern': '{lecture_id}_{date}_{speaker}.srt'
                }
            },
            'directories': {
                'input': 'data/raw_srts',
                'output': 'data/processed_srts'
            },
            'validation': {
                'srt_files': {
                    'max_size_mb': 50,
                    'required_extensions': ['.srt']
                },
                'naming_constraints': {
                    'max_filename_length': 255,
                    'forbidden_chars': ['<', '>', ':', '"', '|', '?', '*']
                }
            }
        }
        
        config_file = self.temp_path / "config" / "file_naming.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_processing_workflow(self):
        """Test complete workflow from ingestion to output."""
        # Change to temp directory
        original_cwd = Path.cwd()
        os.chdir(self.temp_path)
        
        try:
            # Create test input files
            input_files = [
                "VED001_20241201_SwamiBrahmananda.srt",
                "VED002_20241202_SwamiBrahmavidyananda.srt"
            ]
            
            for filename in input_files:
                file_path = self.temp_path / "data" / "raw_srts" / filename
                file_path.write_text(f"Test content for {filename}\n1\n00:00:00,000 --> 00:00:05,000\nTest")
            
            # Initialize systems
            ingestion = FileIngestionSystem("config/file_naming.yaml")
            output_manager = OutputStorageManager("config/file_naming.yaml")
            
            # Discover and validate files
            discovered_files = ingestion.discover_files()
            valid_files = ingestion.get_valid_files()
            
            self.assertEqual(len(discovered_files), 2)
            self.assertEqual(len(valid_files), 2)
            
            # Process each valid file
            processed_metadata = []
            for file_info in valid_files:
                # Simulate processing
                processed_content = f"PROCESSED: {file_info.filepath.read_text()}"
                
                # Store processed file
                output_path, metadata = output_manager.store_processed_file(
                    file_info.filepath,
                    processed_content,
                    {'word_error_rate': 0.05, 'confidence': 0.95}
                )
                
                processed_metadata.append(metadata)
                
                # Verify output file exists
                self.assertTrue(output_path.exists())
                self.assertIn("PROCESSED:", output_path.read_text())
            
            # Create processing manifest
            manifest_path = output_manager.create_processing_manifest(processed_metadata)
            self.assertTrue(manifest_path.exists())
            
            # Verify manifest content
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            self.assertEqual(manifest_data['total_files_processed'], 2)
            self.assertEqual(manifest_data['successful_processings'], 2)
            self.assertEqual(manifest_data['failed_processings'], 0)
            
            # Generate ingestion report
            report = ingestion.generate_ingestion_report()
            self.assertEqual(report['total_files_found'], 2)
            self.assertEqual(report['valid_files'], 2)
            self.assertEqual(report['invalid_files'], 0)
            self.assertEqual(report['validation_success_rate'], 100.0)
            
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    # Add import fix for Windows
    import os
    
    # Run tests
    unittest.main(verbosity=2)