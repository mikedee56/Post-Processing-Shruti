"""
Comprehensive pytest configuration and fixtures for Story 5.5 Testing Framework
Provides centralized test configuration, fixtures, and utilities
"""

import pytest
import tempfile
import logging
import time
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test utilities and configurations
from utils.logger_config import setup_test_logging
from utils.config_manager import ConfigurationManager
from utils.metrics_collector import MetricsCollector
from post_processors.sanskrit_post_processor import SanskritPostProcessor
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier
from utils.srt_parser import SRTParser, SRTSegment


# Test Configuration
@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide centralized test configuration"""
    return {
        "test_data_dir": Path(__file__).parent / "test_data",
        "temp_dir": Path(tempfile.gettempdir()) / "asr_post_processing_tests",
        "log_level": logging.DEBUG,
        "performance_timeout": 30,  # seconds
        "quality_threshold": 0.8,
        "coverage_threshold": 0.90,
        "enable_slow_tests": os.getenv("RUN_SLOW_TESTS", "false").lower() == "true",
        "enable_integration_tests": os.getenv("RUN_INTEGRATION_TESTS", "true").lower() == "true",
        "enable_performance_tests": os.getenv("RUN_PERFORMANCE_TESTS", "false").lower() == "true"
    }


@pytest.fixture(scope="session")
def test_logging(test_config):
    """Setup comprehensive test logging"""
    setup_test_logging(
        log_level=test_config["log_level"],
        log_dir=test_config["temp_dir"] / "logs"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(test_config, test_logging):
    """Setup test environment and cleanup"""
    # Create test directories
    test_config["temp_dir"].mkdir(parents=True, exist_ok=True)
    (test_config["temp_dir"] / "logs").mkdir(exist_ok=True)
    (test_config["temp_dir"] / "data").mkdir(exist_ok=True)
    (test_config["temp_dir"] / "metrics").mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after all tests
    import shutil
    if test_config["temp_dir"].exists():
        shutil.rmtree(test_config["temp_dir"], ignore_errors=True)


# Component Fixtures
@pytest.fixture
def mock_configuration() -> ConfigurationManager:
    """Provide mocked configuration manager for testing"""
    config = ConfigurationManager()
    config.config = {
        "processing": {
            "enable_performance_monitoring": False,
            "enable_logging": False,
            "batch_size": 10,
            "max_workers": 2
        },
        "text_normalization": {
            "enable_mcp_processing": False,
            "enable_monitoring": False,
            "enable_qa": False
        },
        "monitoring": {
            "metrics_enabled": False,
            "tracing_enabled": False,
            "health_checks_enabled": False
        }
    }
    return config


@pytest.fixture
def sanskrit_post_processor(mock_configuration) -> SanskritPostProcessor:
    """Provide Sanskrit post processor for testing"""
    return SanskritPostProcessor(mock_configuration.config)


@pytest.fixture
def sanskrit_hindi_identifier() -> SanskritHindiIdentifier:
    """Provide Sanskrit/Hindi identifier for testing"""
    return SanskritHindiIdentifier()


@pytest.fixture
def srt_parser() -> SRTParser:
    """Provide SRT parser for testing"""
    return SRTParser()


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Provide metrics collector for testing"""
    return MetricsCollector()


# Test Data Fixtures
@pytest.fixture
def sample_srt_content() -> str:
    """Provide sample SRT content for testing"""
    return """1
00:00:01,000 --> 00:00:05,000
Today we study krishna and dharma from the bhagavad gita.

2
00:00:06,000 --> 00:00:10,000
This teaches us about yoga and meditation practices.

3
00:00:11,000 --> 00:00:15,000
We learn about chapter two verse twenty five.
"""


@pytest.fixture
def sample_srt_segment() -> SRTSegment:
    """Provide sample SRT segment for testing"""
    return SRTSegment(
        index=1,
        start_time="00:00:01,000",
        end_time="00:00:05,000",
        text="Today we study krishna and dharma from ancient scriptures.",
        raw_text="Today we study krishna and dharma from ancient scriptures."
    )


@pytest.fixture
def test_srt_file(test_config, sample_srt_content) -> Generator[Path, None, None]:
    """Create temporary SRT file for testing"""
    temp_file = test_config["temp_dir"] / "test_sample.srt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_srt_content)
    
    yield temp_file
    
    # Cleanup
    if temp_file.exists():
        temp_file.unlink()


@pytest.fixture
def performance_test_data() -> Dict[str, Any]:
    """Provide performance testing data and thresholds"""
    return {
        "segments_per_second_target": 10.0,
        "max_processing_time": 5.0,  # seconds
        "memory_usage_limit": 500,   # MB
        "variance_threshold": 10.0,  # percent
        "load_test_segments": 100,
        "stress_test_segments": 1000
    }


# Test Utilities
@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities"""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.measurements = []
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            if self.start_time:
                elapsed = time.perf_counter() - self.start_time
                self.measurements.append(elapsed)
                return elapsed
            return 0
        
        def get_average(self):
            return sum(self.measurements) / len(self.measurements) if self.measurements else 0
        
        def get_variance_percentage(self):
            if len(self.measurements) < 2:
                return 0
            avg = self.get_average()
            variance = sum((x - avg) ** 2 for x in self.measurements) / len(self.measurements)
            return (variance ** 0.5) / avg * 100 if avg > 0 else 0
    
    return PerformanceMonitor()


@pytest.fixture
def quality_validator():
    """Provide quality validation utilities"""
    class QualityValidator:
        def validate_text_normalization(self, original: str, processed: str) -> Dict[str, Any]:
            """Validate text normalization quality"""
            return {
                "length_change": abs(len(processed) - len(original)) / len(original),
                "character_preservation": sum(1 for a, b in zip(original.lower(), processed.lower()) if a == b) / len(original),
                "whitespace_normalized": processed.count(' ') <= original.count(' ') + 5,
                "no_corruption": all(ord(c) < 128 or c.isprintable() for c in processed)
            }
        
        def validate_sanskrit_accuracy(self, text: str, expected_terms: list) -> Dict[str, Any]:
            """Validate Sanskrit term accuracy"""
            found_terms = [term for term in expected_terms if term in text]
            return {
                "terms_found": len(found_terms),
                "terms_expected": len(expected_terms),
                "accuracy": len(found_terms) / len(expected_terms) if expected_terms else 1.0,
                "missing_terms": [term for term in expected_terms if term not in text]
            }
    
    return QualityValidator()


# Parametrized Test Fixtures
@pytest.fixture(params=[
    "simple_text",
    "sanskrit_heavy",
    "number_conversion",
    "conversational_patterns"
])
def test_text_variety(request):
    """Provide variety of test texts for comprehensive testing"""
    test_texts = {
        "simple_text": "This is a simple test text for processing.",
        "sanskrit_heavy": "Today we study Krishna and Dharma from Bhagavad Gita with Arjuna and Rama.",
        "number_conversion": "Chapter two verse twenty five teaches us about two thousand five concepts.",
        "conversational_patterns": "Um, today we will, uh, actually study the teachings, you know."
    }
    return {
        "type": request.param,
        "text": test_texts[request.param]
    }


# Slow Test Marker
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "quality: marks tests as quality validation tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment settings"""
    # Skip slow tests if not enabled
    if not os.getenv("RUN_SLOW_TESTS", "false").lower() == "true":
        skip_slow = pytest.mark.skip(reason="Slow tests disabled (set RUN_SLOW_TESTS=true to enable)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip integration tests if not enabled
    if not os.getenv("RUN_INTEGRATION_TESTS", "true").lower() == "true":
        skip_integration = pytest.mark.skip(reason="Integration tests disabled")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    
    # Skip performance tests if not enabled
    if not os.getenv("RUN_PERFORMANCE_TESTS", "false").lower() == "true":
        skip_performance = pytest.mark.skip(reason="Performance tests disabled")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)