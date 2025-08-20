"""
Test Fixtures and Mocking Infrastructure for Story 5.5: Testing & Quality Assurance Framework

This module provides comprehensive test fixtures, mocking utilities, and test data
generation for the complete ASR post-processing testing framework.
"""

import json
import yaml
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import random


@dataclass
class TestFixture:
    """Represents a test fixture with metadata."""
    name: str
    category: str
    data: Any
    description: str
    dependencies: List[str]
    cleanup_required: bool = False


class TestFixtureManager:
    """
    Comprehensive test fixture manager for the ASR post-processing testing framework.
    
    Provides fixtures for SRT files, Sanskrit/Hindi text, processing configurations,
    mock objects, and test environments.
    """
    
    def __init__(self):
        """Initialize the test fixture manager."""
        self.fixtures: Dict[str, TestFixture] = {}
        self.temp_directories: List[Path] = []
        self.mock_objects: Dict[str, Mock] = {}
        
        # Initialize default fixtures
        self._initialize_default_fixtures()
    
    def _initialize_default_fixtures(self):
        """Initialize default test fixtures."""
        
        # SRT Content Fixtures
        self.register_fixture(
            "basic_srt_content",
            "srt_data",
            """1
00:00:01,000 --> 00:00:05,000
Today we study yoga and dharma.

2
00:00:06,000 --> 00:00:10,000
Krishna teaches us about moksha.""",
            "Basic SRT content for simple testing"
        )
        
        self.register_fixture(
            "complex_srt_content",
            "srt_data",
            """1
00:00:01,000 --> 00:00:05,000
today we study krsna in bhagavad geeta chapter two verse twenty five.

2
00:00:06,000 --> 00:00:10,000
um, this verse, uh, teaches about dharama and, you know, yoga practices.

3
00:00:11,000 --> 00:00:15,000
In the year two thousand five, swami vivekananda explained this.

4
00:00:16,000 --> 00:00:20,000
patanjalee wrote the yog sutras about, actually, eight limbs of yog.""",
            "Complex SRT content with multiple processing challenges"
        )
        
        self.register_fixture(
            "scriptural_srt_content",
            "srt_data",
            """1
00:00:01,000 --> 00:00:05,000
The bhagvad gita chapter two verse twenty five says avyakto yam acintyo yam.

2
00:00:06,000 --> 00:00:10,000
yoga sutras of patanjalee explain the eight fold path.

3
00:00:11,000 --> 00:00:15,000
upanishads teach us about brahman and atman.""",
            "SRT content focused on scriptural references"
        )
        
        # Sanskrit/Hindi Term Fixtures
        self.register_fixture(
            "sanskrit_terms_basic",
            "language_data",
            {
                "basic_terms": ["krishna", "dharma", "yoga", "moksha", "karma"],
                "complex_terms": ["pranayama", "dharana", "dhyana", "samadhi", "ahimsa"],
                "proper_nouns": ["Krishna", "Shiva", "Vishnu", "Patanjali", "Shankaracharya"],
                "scriptures": ["Bhagavad Gita", "Yoga Sutras", "Upanishads", "Ramayana"],
                "places": ["Rishikesh", "Varanasi", "Haridwar", "Vrindavan"]
            },
            "Basic Sanskrit/Hindi terms for testing"
        )
        
        # Processing Configuration Fixtures
        self.register_fixture(
            "test_configurations",
            "config_data",
            {
                "performance": {
                    "target_segments_per_second": 10.0,
                    "variance_threshold_percent": 10.0,
                    "memory_limit_mb": 500,
                    "timeout_seconds": 30
                },
                "accuracy": {
                    "minimum_accuracy_threshold": 0.90,
                    "word_error_rate_threshold": 0.1,
                    "character_error_rate_threshold": 0.05
                },
                "quality": {
                    "coverage_threshold": 0.95,
                    "complexity_score_threshold": 0.8
                }
            },
            "Test configurations for performance and quality validation"
        )
        
        # Mock Response Fixtures
        self.register_fixture(
            "mock_responses",
            "mock_data",
            {
                "mcp_client": {
                    "success": {"status": "success", "confidence": 0.95, "processing_time": 0.1},
                    "failure": {"status": "error", "message": "Processing failed", "error_code": "PROC_001"},
                    "timeout": {"status": "timeout", "message": "Request timed out", "retry_after": 5}
                },
                "api_responses": {
                    "text_processing": {
                        "success": {
                            "original_text": "today we study krsna",
                            "corrected_text": "Today we study Krishna",
                            "corrections_applied": ["capitalization", "sanskrit_correction"],
                            "confidence": 0.98
                        }
                    }
                }
            },
            "Mock API responses for testing"
        )
    
    def register_fixture(
        self, 
        name: str, 
        category: str, 
        data: Any, 
        description: str,
        dependencies: Optional[List[str]] = None,
        cleanup_required: bool = False
    ):
        """Register a new test fixture."""
        
        fixture = TestFixture(
            name=name,
            category=category,
            data=data,
            description=description,
            dependencies=dependencies or [],
            cleanup_required=cleanup_required
        )
        
        self.fixtures[name] = fixture
    
    def get_fixture(self, name: str) -> Any:
        """Get fixture data by name."""
        if name not in self.fixtures:
            raise ValueError(f"Fixture '{name}' not found")
        
        return self.fixtures[name].data
    
    def create_temp_srt_file(self, content: str, filename: Optional[str] = None) -> Path:
        """Create a temporary SRT file with given content."""
        
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_directories.append(temp_dir)
        
        if filename is None:
            filename = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
        
        srt_path = temp_dir / filename
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return srt_path
    
    def create_temp_config_file(self, config: Dict[str, Any], format: str = "yaml") -> Path:
        """Create a temporary configuration file."""
        
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_directories.append(temp_dir)
        
        if format == "yaml":
            config_path = temp_dir / "test_config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif format == "json":
            config_path = temp_dir / "test_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return config_path
    
    def create_mock_processor(self, behavior: str = "success") -> Mock:
        """Create a mock SanskritPostProcessor with configurable behavior."""
        
        mock_processor = Mock()
        
        if behavior == "success":
            mock_processor.process_srt_file.return_value = Mock(
                total_segments=5,
                segments_modified=3,
                processing_time=0.5,
                average_confidence=0.95
            )
            mock_processor.text_normalizer.normalize_with_advanced_tracking.return_value = Mock(
                corrected_text="Test output",
                corrections_applied=["test_correction"],
                quality_score=0.9
            )
        elif behavior == "failure":
            mock_processor.process_srt_file.side_effect = Exception("Processing failed")
        elif behavior == "timeout":
            import time
            def slow_process(*args, **kwargs):
                time.sleep(10)  # Simulate timeout
                return Mock()
            mock_processor.process_srt_file.side_effect = slow_process
        
        self.mock_objects[f"mock_processor_{behavior}"] = mock_processor
        return mock_processor
    
    def create_mock_mcp_client(self, behavior: str = "success") -> Mock:
        """Create a mock MCP client with configurable behavior."""
        
        mock_client = Mock()
        
        if behavior == "success":
            mock_client.get_performance_stats.return_value = {
                "requests_processed": 100,
                "average_response_time": 0.1,
                "error_rate": 0.02
            }
            mock_client.process_text.return_value = {
                "status": "success",
                "result": "processed text",
                "confidence": 0.95
            }
        elif behavior == "connection_error":
            mock_client.get_performance_stats.side_effect = ConnectionError("Cannot connect to MCP server")
        elif behavior == "api_error":
            mock_client.process_text.side_effect = Exception("API processing error")
        
        self.mock_objects[f"mock_mcp_client_{behavior}"] = mock_client
        return mock_client
    
    def generate_test_data_variants(self, base_text: str, variant_count: int = 10) -> List[str]:
        """Generate variants of test data for comprehensive testing."""
        
        variants = [base_text]  # Include original
        
        # Common ASR errors and variations
        error_patterns = [
            ("krishna", "krsna"),
            ("dharma", "dharama"),
            ("yoga", "yog"),
            ("chapter", "chaptor"),
            ("verse", "vers"),
            ("two", "too"),
            ("twenty", "20"),
            ("Bhagavad Gita", "bhagvad geeta"),
            ("Patanjali", "patanjalee")
        ]
        
        for i in range(min(variant_count - 1, len(error_patterns))):
            variant = base_text
            original, corrupted = error_patterns[i]
            if original.lower() in variant.lower():
                variant = variant.replace(original, corrupted)
                variant = variant.replace(original.capitalize(), corrupted.capitalize())
                variants.append(variant)
        
        # Generate additional random variants
        while len(variants) < variant_count:
            variant = base_text
            
            # Add filler words occasionally
            if random.random() < 0.3:
                words = variant.split()
                insert_pos = random.randint(0, len(words))
                filler = random.choice(["um,", "uh,", "you know,", "actually,"])
                words.insert(insert_pos, filler)
                variant = " ".join(words)
            
            # Apply random capitalization changes
            if random.random() < 0.5:
                variant = variant.lower()
            
            variants.append(variant)
        
        return variants[:variant_count]
    
    def cleanup_temp_data(self):
        """Clean up all temporary directories and files."""
        
        for temp_dir in self.temp_directories:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up {temp_dir}: {e}")
        
        self.temp_directories.clear()
        self.mock_objects.clear()


# Global fixture manager instance
_fixture_manager = TestFixtureManager()


# Pytest fixtures for integration with pytest framework
@pytest.fixture(scope="session")
def fixture_manager():
    """Pytest fixture providing the test fixture manager."""
    return _fixture_manager


@pytest.fixture(scope="function")
def temp_srt_file(fixture_manager):
    """Pytest fixture providing a temporary SRT file."""
    content = fixture_manager.get_fixture("basic_srt_content")
    return fixture_manager.create_temp_srt_file(content)


@pytest.fixture(scope="function")
def complex_srt_file(fixture_manager):
    """Pytest fixture providing a complex temporary SRT file."""
    content = fixture_manager.get_fixture("complex_srt_content")
    return fixture_manager.create_temp_srt_file(content)


@pytest.fixture(scope="function")
def sanskrit_terms(fixture_manager):
    """Pytest fixture providing Sanskrit/Hindi terms."""
    return fixture_manager.get_fixture("sanskrit_terms_basic")


@pytest.fixture(scope="function")
def test_config(fixture_manager):
    """Pytest fixture providing test configuration."""
    return fixture_manager.get_fixture("test_configurations")


@pytest.fixture(scope="function")
def mock_processor_success(fixture_manager):
    """Pytest fixture providing a successful mock processor."""
    return fixture_manager.create_mock_processor("success")


@pytest.fixture(scope="function")
def mock_processor_failure(fixture_manager):
    """Pytest fixture providing a failing mock processor."""
    return fixture_manager.create_mock_processor("failure")


@pytest.fixture(scope="function")
def mock_mcp_client(fixture_manager):
    """Pytest fixture providing a mock MCP client."""
    return fixture_manager.create_mock_mcp_client("success")


@pytest.fixture(scope="function", autouse=True)
def cleanup_fixtures(fixture_manager):
    """Pytest fixture for automatic cleanup after each test."""
    yield
    fixture_manager.cleanup_temp_data()


# Context managers for test isolation
class MockEnvironment:
    """Context manager for isolated mock testing environment."""
    
    def __init__(self, fixture_manager: TestFixtureManager, mocks: Dict[str, str]):
        self.fixture_manager = fixture_manager
        self.mocks = mocks
        self.patches = []
    
    def __enter__(self):
        """Enter the mock environment."""
        
        # Apply patches based on mock configuration
        for target, behavior in self.mocks.items():
            if "processor" in target.lower():
                mock_obj = self.fixture_manager.create_mock_processor(behavior)
                patch_obj = patch(target, return_value=mock_obj)
            elif "mcp" in target.lower():
                mock_obj = self.fixture_manager.create_mock_mcp_client(behavior)
                patch_obj = patch(target, return_value=mock_obj)
            else:
                mock_obj = Mock()
                patch_obj = patch(target, mock_obj)
            
            self.patches.append(patch_obj)
            patch_obj.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the mock environment."""
        
        for patch_obj in self.patches:
            patch_obj.stop()
        
        self.fixture_manager.cleanup_temp_data()


# Utility functions for test data generation
def generate_edge_case_srt_content() -> List[str]:
    """Generate edge case SRT content for comprehensive testing."""
    
    edge_cases = [
        # Empty content
        "",
        
        # Malformed timestamps
        """1
invalid_timestamp
Text without proper timestamps.""",
        
        # Missing sequence numbers
        """
00:00:01,000 --> 00:00:05,000
Text without sequence number.""",
        
        # Extremely long content
        """1
00:00:01,000 --> 00:00:05,000
""" + "Very long content " * 100,
        
        # Special characters and Unicode
        """1
00:00:01,000 --> 00:00:05,000
Text with special chars: @#$%^&*() and unicode: āīūṛṃḥ""",
        
        # Mixed languages
        """1
00:00:01,000 --> 00:00:05,000
Today हम study करते हैं yoga और dharma."""
    ]
    
    return edge_cases


def create_performance_test_data(segment_count: int = 100) -> str:
    """Create SRT content for performance testing."""
    
    template = """{}
00:00:{:02d},000 --> 00:00:{:02d},500
Today we study {} and {} from the ancient texts."""
    
    sanskrit_terms = ["yoga", "dharma", "moksha", "karma", "ahimsa", "pranayama"]
    
    srt_content = []
    
    for i in range(segment_count):
        start_time = i
        end_time = i + 1
        term1 = random.choice(sanskrit_terms)
        term2 = random.choice(sanskrit_terms)
        
        segment = template.format(i + 1, start_time, end_time, term1, term2)
        srt_content.append(segment)
    
    return "\n\n".join(srt_content)


# Integration testing utilities
def setup_test_environment() -> Dict[str, Any]:
    """Set up a complete test environment for integration testing."""
    
    fixture_manager = TestFixtureManager()
    
    # Create test data directory structure
    test_data_dir = Path("tests/data/test_environment")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample SRT files
    srt_files = {}
    for fixture_name in ["basic_srt_content", "complex_srt_content", "scriptural_srt_content"]:
        content = fixture_manager.get_fixture(fixture_name)
        srt_path = fixture_manager.create_temp_srt_file(content, f"{fixture_name}.srt")
        srt_files[fixture_name] = srt_path
    
    # Create configuration files
    config = fixture_manager.get_fixture("test_configurations")
    config_path = fixture_manager.create_temp_config_file(config)
    
    return {
        "fixture_manager": fixture_manager,
        "test_data_dir": test_data_dir,
        "srt_files": srt_files,
        "config_path": config_path,
        "sanskrit_terms": fixture_manager.get_fixture("sanskrit_terms_basic")
    }


if __name__ == "__main__":
    # Demonstrate fixture system
    print("Test Fixtures and Mocking Infrastructure")
    print("=" * 50)
    
    # Initialize fixture manager
    manager = TestFixtureManager()
    
    # Show available fixtures
    print(f"Available fixtures: {len(manager.fixtures)}")
    for name, fixture in manager.fixtures.items():
        print(f"  {name} ({fixture.category}): {fixture.description}")
    
    # Create temporary test data
    srt_content = manager.get_fixture("complex_srt_content")
    temp_file = manager.create_temp_srt_file(srt_content)
    print(f"\nCreated temporary SRT file: {temp_file}")
    
    # Generate test variants
    base_text = "Today we study Krishna and dharma"
    variants = manager.generate_test_data_variants(base_text, 5)
    print(f"\nGenerated {len(variants)} test variants:")
    for i, variant in enumerate(variants, 1):
        print(f"  {i}. {variant}")
    
    # Create mock objects
    mock_processor = manager.create_mock_processor("success")
    print(f"\nCreated mock processor: {type(mock_processor)}")
    
    # Cleanup
    manager.cleanup_temp_data()
    print("\nCleanup completed")