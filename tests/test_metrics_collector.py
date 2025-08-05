"""
Unit tests for Metrics Collector functionality.

Tests comprehensive metrics collection, session management,
and reporting functionality.
"""

import pytest
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.utils.metrics_collector import (
    MetricsCollector, ProcessingMetrics, SessionMetrics
)


class TestProcessingMetrics:
    """Test suite for ProcessingMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating ProcessingMetrics instances."""
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.5,
            total_segments=10,
            segments_modified=5
        )
        
        assert metrics.file_path == "test.srt"
        assert metrics.processing_time == 1.5
        assert metrics.total_segments == 10
        assert metrics.segments_modified == 5
        assert metrics.corrections_applied == {}
        assert metrics.timestamp_integrity_verified is True
        assert metrics.errors_encountered == []
    
    def test_metrics_with_optional_fields(self):
        """Test creating ProcessingMetrics with optional fields."""
        corrections = {"number_conversion": 3, "filler_removal": 2}
        errors = ["test error"]
        
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=2.0,
            total_segments=15,
            segments_modified=8,
            corrections_applied=corrections,
            errors_encountered=errors
        )
        
        assert metrics.corrections_applied == corrections
        assert metrics.errors_encountered == errors


class TestSessionMetrics:
    """Test suite for SessionMetrics dataclass."""
    
    def test_session_creation(self):
        """Test creating SessionMetrics instances."""
        session = SessionMetrics(
            session_id="test_session",
            start_time="2025-01-01T00:00:00Z"
        )
        
        assert session.session_id == "test_session"
        assert session.start_time == "2025-01-01T00:00:00Z"
        assert session.end_time is None
        assert session.total_files_processed == 0
        assert session.file_metrics == []


class TestMetricsCollector:
    """Test suite for MetricsCollector class."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Use temporary directory for metrics
        self.temp_dir = tempfile.mkdtemp()
        config = {
            'metrics_dir': self.temp_dir,
            'auto_save': False  # Disable auto-save for tests
        }
        self.collector = MetricsCollector(config)
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_file_metrics(self):
        """Test creating file metrics."""
        metrics = self.collector.create_file_metrics("test.srt")
        
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.file_path == "test.srt"
        assert metrics.processing_time == 0.0
        assert metrics.total_segments == 0
    
    def test_start_end_session(self):
        """Test starting and ending sessions."""
        # Start session
        session_id = self.collector.start_session("test_session")
        assert session_id == "test_session"
        assert self.collector.current_session is not None
        assert self.collector.current_session.session_id == "test_session"
        
        # End session
        final_session = self.collector.end_session()
        assert final_session is not None
        assert final_session.session_id == "test_session"
        assert final_session.end_time is not None
        assert self.collector.current_session is None
    
    def test_auto_session_id(self):
        """Test automatic session ID generation."""
        session_id = self.collector.start_session()
        assert session_id.startswith("session_")
        assert self.collector.current_session.session_id == session_id
    
    def test_end_session_without_start(self):
        """Test ending session without starting one."""
        result = self.collector.end_session()
        assert result is None
    
    def test_timer_operations(self):
        """Test timer start/end operations."""
        # Start timer
        self.collector.start_timer("test_operation")
        assert "test_operation" in self.collector.operation_timers
        
        # Wait a bit
        time.sleep(0.01)
        
        # End timer
        elapsed = self.collector.end_timer("test_operation")
        assert elapsed > 0
        assert elapsed < 1.0  # Should be small
        assert "test_operation" not in self.collector.operation_timers
    
    def test_end_timer_not_started(self):
        """Test ending timer that wasn't started."""
        elapsed = self.collector.end_timer("nonexistent")
        assert elapsed == 0.0
    
    def test_add_file_metrics(self):
        """Test adding file metrics to session."""
        # Start session
        self.collector.start_session("test_session")
        
        # Create and add metrics
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5
        )
        
        self.collector.add_file_metrics(metrics)
        
        # Check session was updated
        session = self.collector.current_session
        assert len(session.file_metrics) == 1
        assert session.total_files_processed == 1
        assert session.successful_files == 1
        assert session.failed_files == 0
    
    def test_add_file_metrics_with_errors(self):
        """Test adding file metrics with errors."""
        # Start session
        self.collector.start_session("test_session")
        
        # Create metrics with errors
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5,
            errors_encountered=["test error"]
        )
        
        self.collector.add_file_metrics(metrics)
        
        # Check session counted as failed
        session = self.collector.current_session
        assert session.successful_files == 0
        assert session.failed_files == 1
        assert "test error" in session.session_errors
    
    def test_add_file_metrics_without_session(self):
        """Test adding file metrics without active session."""
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5
        )
        
        # Should auto-create session
        self.collector.add_file_metrics(metrics)
        assert self.collector.current_session is not None
        assert len(self.collector.current_session.file_metrics) == 1
    
    def test_update_correction_count(self):
        """Test updating correction counts."""
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5
        )
        
        # Update correction count
        self.collector.update_correction_count(metrics, "number_conversion", 3)
        assert metrics.corrections_applied["number_conversion"] == 3
        
        # Update same type again
        self.collector.update_correction_count(metrics, "number_conversion", 2)
        assert metrics.corrections_applied["number_conversion"] == 5
        
        # Add different type
        self.collector.update_correction_count(metrics, "filler_removal")
        assert metrics.corrections_applied["filler_removal"] == 1
    
    def test_calculate_quality_metrics(self):
        """Test calculating quality metrics."""
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5,
            confidence_scores=[0.8, 0.9, 0.7, 0.85, 0.75]
        )
        
        self.collector.calculate_quality_metrics(metrics)
        
        # Should calculate average confidence
        expected_avg = sum([0.8, 0.9, 0.7, 0.85, 0.75]) / 5
        assert abs(metrics.average_confidence - expected_avg) < 0.001
    
    def test_calculate_quality_metrics_empty(self):
        """Test calculating quality metrics with no confidence scores."""
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5
        )
        
        self.collector.calculate_quality_metrics(metrics)
        assert metrics.average_confidence == 0.0
    
    def test_generate_processing_report(self):
        """Test generating processing report."""
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=2.5,
            total_segments=20,
            segments_modified=10,
            corrections_applied={"number_conversion": 5, "filler_removal": 3},
            original_word_count=100,
            processed_word_count=95,
            average_confidence=0.85,
            flagged_segments=2
        )
        
        report = self.collector.generate_processing_report(metrics)
        
        # Check report structure
        assert "file_summary" in report
        assert "text_statistics" in report
        assert "corrections" in report
        assert "quality_metrics" in report
        assert "performance" in report
        assert "issues" in report
        
        # Check specific values
        assert report["file_summary"]["file_path"] == "test.srt"
        assert report["file_summary"]["total_segments"] == 20
        assert report["corrections"]["total_corrections"] == 8
        assert "0.85" in report["quality_metrics"]["average_confidence"]
    
    def test_generate_session_report(self):
        """Test generating session report."""
        # Create session with some metrics
        self.collector.start_session("test_session")
        
        metrics1 = ProcessingMetrics(
            file_path="test1.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5,
            corrections_applied={"number_conversion": 2},
            average_confidence=0.8
        )
        
        metrics2 = ProcessingMetrics(
            file_path="test2.srt",
            processing_time=1.5,
            total_segments=15,
            segments_modified=8,
            corrections_applied={"filler_removal": 3},
            average_confidence=0.9
        )
        
        self.collector.add_file_metrics(metrics1)
        self.collector.add_file_metrics(metrics2)
        
        session = self.collector.end_session()
        report = self.collector.generate_session_report(session)
        
        # Check report structure
        assert "session_summary" in report
        assert "processing_statistics" in report
        assert "performance" in report
        assert "corrections_by_type" in report
        assert "file_results" in report
        
        # Check aggregated values
        assert report["processing_statistics"]["total_segments"] == 25
        assert report["processing_statistics"]["total_modified_segments"] == 13
        assert report["corrections_by_type"]["number_conversion"] == 2
        assert report["corrections_by_type"]["filler_removal"] == 3
        assert len(report["file_results"]) == 2
    
    def test_generate_session_report_no_session(self):
        """Test generating session report with no session."""
        report = self.collector.generate_session_report()
        assert "error" in report
    
    def test_save_load_session_metrics(self):
        """Test saving and loading session metrics."""
        # Create session with metrics
        self.collector.start_session("test_session")
        
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5
        )
        
        self.collector.add_file_metrics(metrics)
        session = self.collector.end_session()
        
        # Save session
        save_path = self.collector.save_session_metrics(session)
        assert save_path.exists()
        
        # Load session
        loaded_session = self.collector.load_session_metrics("test_session")
        
        assert loaded_session is not None
        assert loaded_session.session_id == "test_session"
        assert loaded_session.total_files_processed == 1
        assert len(loaded_session.file_metrics) == 1
    
    def test_load_nonexistent_session(self):
        """Test loading non-existent session."""
        result = self.collector.load_session_metrics("nonexistent")
        assert result is None
    
    def test_save_file_metrics(self):
        """Test saving individual file metrics."""
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5
        )
        
        # Enable individual file saving
        self.collector.save_individual_files = True
        save_path = self.collector.save_file_metrics(metrics)
        
        assert save_path.exists()
        assert "test_metrics_" in save_path.name
        
        # Verify content
        with open(save_path, 'r') as f:
            data = json.load(f)
        
        assert data["file_path"] == "test.srt"
        assert data["total_segments"] == 10
    
    def test_metrics_with_auto_save(self):
        """Test metrics collection with auto-save enabled."""
        # Create collector with auto-save
        config = {
            'metrics_dir': self.temp_dir,
            'auto_save': True,
            'save_individual_files': True
        }
        collector = MetricsCollector(config)
        
        # Create session and add metrics
        collector.start_session("auto_save_test")
        
        metrics = ProcessingMetrics(
            file_path="test.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=5
        )
        
        collector.add_file_metrics(metrics)
        session = collector.end_session()
        
        # Check that files were auto-saved
        session_file = Path(self.temp_dir) / f"{session.session_id}_metrics.json"
        assert session_file.exists()
    
    def test_session_timing(self):
        """Test session timing calculations."""
        self.collector.start_session("timing_test")
        
        # Wait a bit
        time.sleep(0.01)
        
        session = self.collector.end_session()
        
        assert session.total_processing_time > 0
        assert session.total_processing_time < 1.0  # Should be small
    
    def test_metrics_directory_creation(self):
        """Test that metrics directory is created if it doesn't exist."""
        nonexistent_dir = Path(self.temp_dir) / "new_metrics_dir"
        assert not nonexistent_dir.exists()
        
        config = {'metrics_dir': str(nonexistent_dir)}
        collector = MetricsCollector(config)
        
        assert nonexistent_dir.exists()
        assert nonexistent_dir.is_dir()


class TestMetricsIntegration:
    """Integration tests for metrics collection workflow."""
    
    def setup_method(self):
        """Setup for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        config = {
            'metrics_dir': self.temp_dir,
            'auto_save': True,
            'save_individual_files': True
        }
        self.collector = MetricsCollector(config)
    
    def teardown_method(self):
        """Cleanup after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_processing_workflow(self):
        """Test complete processing workflow with metrics."""
        # Start session
        session_id = self.collector.start_session("workflow_test")
        
        # Process multiple files
        for i in range(3):
            # Create metrics for file
            metrics = self.collector.create_file_metrics(f"file_{i}.srt")
            
            # Simulate processing with timing
            self.collector.start_timer("parsing")
            time.sleep(0.001)  # Simulate parsing time
            metrics.parsing_time = self.collector.end_timer("parsing")
            
            self.collector.start_timer("normalization")
            time.sleep(0.001)  # Simulate normalization time
            metrics.normalization_time = self.collector.end_timer("normalization")
            
            # Set metrics values
            metrics.total_segments = 10 + i * 5
            metrics.segments_modified = 5 + i * 2
            metrics.confidence_scores = [0.8 + i * 0.05] * metrics.total_segments
            
            # Update correction counts
            self.collector.update_correction_count(metrics, "number_conversion", i + 1)
            self.collector.update_correction_count(metrics, "filler_removal", i * 2)
            
            # Calculate quality metrics
            self.collector.calculate_quality_metrics(metrics)
            
            # Add to session
            self.collector.add_file_metrics(metrics)
        
        # End session
        final_session = self.collector.end_session()
        
        # Verify session results
        assert final_session.total_files_processed == 3
        assert final_session.successful_files == 3
        assert final_session.failed_files == 0
        
        # Verify aggregations
        assert final_session.total_segments_processed == 45  # 10+15+20
        assert final_session.total_corrections_applied == 9  # (1+2+3) + (0+2+4)
        
        # Verify files were saved
        session_file = Path(self.temp_dir) / f"{session_id}_metrics.json"
        assert session_file.exists()
        
        # Verify individual file metrics were saved
        metrics_files = list(Path(self.temp_dir).glob("file_*_metrics_*.json"))
        assert len(metrics_files) == 3
    
    def test_error_handling_workflow(self):
        """Test workflow with errors."""
        self.collector.start_session("error_test")
        
        # Create metrics with errors
        metrics = ProcessingMetrics(
            file_path="error_file.srt",
            processing_time=1.0,
            total_segments=10,
            segments_modified=0,
            errors_encountered=["Parse error", "Validation error"]
        )
        
        self.collector.add_file_metrics(metrics)
        session = self.collector.end_session()
        
        # Should be counted as failed
        assert session.failed_files == 1
        assert session.successful_files == 0
        assert len(session.session_errors) == 2
        
        # Generate report
        report = self.collector.generate_session_report(session)
        assert report["session_summary"]["success_rate"] == "0.0%"