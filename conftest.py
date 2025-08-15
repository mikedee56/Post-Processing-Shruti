#!/usr/bin/env python3
"""
Pytest configuration file for automatic test setup.
Handles path configuration and test environment setup for Story 4.5.
"""

import sys
import os
from pathlib import Path

def pytest_configure(config):
    """Configure pytest with proper path setup."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    src_path = project_root / "src"
    
    # Add src directory to Python path for clean imports
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set environment variable for consistent path handling
    os.environ['PYTHONPATH'] = str(src_path)
    
    print(f"Test environment configured with src path: {src_path}")

def pytest_sessionstart(session):
    """Session start hook for additional setup."""
    
    # Verify that critical modules can be imported
    try:
        # Test Story 2.3 compatibility
        from scripture_processing.scripture_processor import ScriptureProcessor
        
        # Test Story 4.5 components
        from scripture_processing.advanced_verse_matcher import AdvancedVerseMatcher
        from scripture_processing.academic_citation_manager import AcademicCitationManager
        from scripture_processing.publication_formatter import PublicationFormatter
        from utils.academic_validator import AcademicValidator
        
        print("All critical modules import successfully in test environment")
        
    except ImportError as e:
        print(f"Warning: Some modules may not be available during testing: {e}")
        # Don't fail the session, just warn

def pytest_collection_modifyitems(config, items):
    """Modify collected test items for better organization."""
    
    # Add markers for different test categories
    for item in items:
        # Mark Story 4.5 specific tests
        if "story_4_5" in item.nodeid or "task_" in item.nodeid:
            item.add_marker("story_4_5")
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker("integration")
        
        # Mark validation tests
        if "validation" in item.nodeid:
            item.add_marker("validation")