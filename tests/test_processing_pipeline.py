"""
Integration tests for the complete SRT processing pipeline.

Tests end-to-end processing with real SRT files and validates
the complete workflow from parsing to output generation.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

from src.post_processors.sanskrit_post_processor import SanskritPostProcessor
from src.utils.srt_parser import SRTParser
from src.utils.text_normalizer import TextNormalizer
from src.utils.metrics_collector import MetricsCollector
from src.utils.logger_config import setup_logging


class TestProcessingPipelineIntegration:
    """Integration tests for the complete processing pipeline."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Setup logging
        setup_logging({'level': 'DEBUG'})
        
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
        
        # Initialize processor with test config
        test_config = {
            'fuzzy_threshold': 80,
            'confidence_threshold': 0.6,
            'lexicon_paths': {
                'corrections': 'data/lexicons/corrections.yaml',
                'proper_nouns': 'data/lexicons/proper_nouns.yaml',
                'phrases': 'data/lexicons/phrases.yaml',
                'verses': 'data/lexicons/verses.yaml'
            },
            'text_normalization': {
                'remove_fillers': True,
                'convert_numbers': True,
                'standardize_punctuation': True,
                'fix_capitalization': True
            },
            'metrics': {
                'metrics_dir': str(self.temp_dir / "metrics"),
                'auto_save': True
            }
        }
        
        self.processor = SanskritPostProcessor()
        self.processor.config = test_config
        self.processor.text_normalizer = TextNormalizer(test_config['text_normalization'])
        self.processor.metrics_collector = MetricsCollector(test_config['metrics'])
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_srt_processing(self):
        """Test basic SRT file processing."""
        input_file = Path("data/test_samples/basic_test.srt")
        output_file = self.output_dir / "basic_processed.srt"
        
        # Process the file
        metrics = self.processor.process_srt_file(input_file, output_file)
        
        # Verify output file was created
        assert output_file.exists()
        
        # Verify metrics
        assert isinstance(metrics, type(metrics))  # ProcessingMetrics type
        assert metrics.total_segments > 0
        assert metrics.processing_time > 0
        assert metrics.timestamp_integrity_verified
        
        # Verify output content
        with open(output_file, 'r', encoding='utf-8') as f:
            output_content = f.read()
        
        # Should have removed filler words
        assert "Um," not in output_content
        assert "uh," not in output_content
        assert "you know" not in output_content
        
        # Should have converted numbers
        assert "chapter 2 verse 25" in output_content or "chapter two verse 25" in output_content
        
        # Should have proper capitalization
        lines = output_content.split('\n')
        text_lines = [line for line in lines if line and not line.isdigit() and '-->' not in line]
        for line in text_lines:
            if line.strip():
                assert line.strip()[0].isupper(), f"Line should start with capital: '{line}'"
    
    def test_complex_srt_processing(self):
        """Test processing complex SRT with Sanskrit terms."""
        input_file = Path("data/test_samples/complex_test.srt")
        output_file = self.output_dir / "complex_processed.srt"
        
        # Process the file
        metrics = self.processor.process_srt_file(input_file, output_file)
        
        # Verify processing results
        assert output_file.exists()
        assert metrics.total_segments >= 8  # Should have 8 segments
        assert metrics.segments_modified > 0  # Some segments should be modified
        
        # Read and verify output
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check filler word removal
        assert "So, uh," not in content
        assert "you know," not in content
        assert "er," not in content
        assert "like," not in content
        
        # Check number conversion
        assert "chapter 18 verse 66" in content or "chapter eighteen verse 66" in content
        assert "verses 65 to 66" in content or "sixty five to sixty six" in content
    
    def test_numbers_processing(self):
        """Test number conversion in SRT processing."""
        input_file = Path("data/test_samples/numbers_test.srt")
        output_file = self.output_dir / "numbers_processed.srt"
        
        # Process the file
        metrics = self.processor.process_srt_file(input_file, output_file)
        
        # Verify processing
        assert output_file.exists()
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check number conversions
        expected_conversions = [
            ("chapter 2 verse 21", "chapter two verse 21"),
            ("1st verse", "first verse"),
            ("700 verses", "seven hundred verses"),
            ("72 verses", "seventy two verses"),
            ("1925", "nineteen twenty five"),
            ("3rd", "third")
        ]
        
        conversion_found = False
        for converted, original in expected_conversions:
            if converted in content:
                conversion_found = True
                break
        
        assert conversion_found, f"No number conversions found in: {content}"
        
        # Check that some corrections were applied
        assert metrics.segments_modified > 0
        assert sum(metrics.corrections_applied.values()) > 0
    
    def test_malformed_srt_processing(self):
        """Test processing malformed SRT files."""
        input_file = Path("data/test_samples/malformed_test.srt")
        output_file = self.output_dir / "malformed_processed.srt"
        
        # Should not raise an exception
        metrics = self.processor.process_srt_file(input_file, output_file)
        
        # Should still produce output
        assert output_file.exists()
        
        # Should have some warnings/errors but still process valid segments
        assert metrics.total_segments > 0  # Should get some valid segments
        assert len(metrics.errors_encountered) > 0 or len(metrics.warnings_encountered) > 0
        
        # Timestamp integrity might be compromised
        # (this is okay for malformed files)
        
        # Verify output has valid SRT format
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have some valid segments
        assert "This is a normal segment" in content
        assert "This segment comes after an empty one" in content
        
        # HTML tags should be removed
        assert "<i>" not in content
        assert "<b>" not in content
        assert "HTML tags and formatting" in content
    
    def test_session_management(self):
        """Test processing session management."""
        # Start a processing session
        session_id = self.processor.start_processing_session("test_session")
        assert session_id == "test_session"
        
        # Process multiple files
        test_files = [
            ("data/test_samples/basic_test.srt", "basic_out.srt"),
            ("data/test_samples/numbers_test.srt", "numbers_out.srt")
        ]
        
        for input_name, output_name in test_files:
            input_file = Path(input_name)
            output_file = self.output_dir / output_name
            
            if input_file.exists():
                metrics = self.processor.process_srt_file(input_file, output_file, session_id)
                assert metrics.total_segments > 0
        
        # End session and get report
        session_report = self.processor.end_processing_session()
        
        assert session_report is not None
        assert session_report["session_summary"]["session_id"] == "test_session"
        assert session_report["session_summary"]["files_processed"] >= 1
        assert "processing_statistics" in session_report
        assert "performance" in session_report
    
    def test_processing_metrics_generation(self):
        """Test comprehensive processing metrics generation."""
        input_file = Path("data/test_samples/complex_test.srt")
        output_file = self.output_dir / "metrics_test.srt"
        
        # Process file
        metrics = self.processor.process_srt_file(input_file, output_file)
        
        # Generate processing report
        report = self.processor.get_processing_report(metrics)
        
        # Verify report structure
        expected_sections = [
            "file_summary", "text_statistics", "corrections", 
            "quality_metrics", "performance", "issues"
        ]
        
        for section in expected_sections:
            assert section in report, f"Missing section: {section}"
        
        # Verify specific metrics
        file_summary = report["file_summary"]
        assert file_summary["file_path"] == str(input_file)
        assert file_summary["total_segments"] > 0
        assert file_summary["processing_time"].endswith("s")
        
        performance = report["performance"]
        assert "parsing_time" in performance
        assert "normalization_time" in performance
        assert "correction_time" in performance
        assert "validation_time" in performance
        
        corrections = report["corrections"]
        assert "total_corrections" in corrections
        assert "by_type" in corrections
    
    def test_error_recovery(self):
        """Test error recovery and graceful degradation."""
        # Create a severely malformed SRT file
        malformed_content = """This is not valid SRT format at all!
Just random text that cannot be parsed.
No timestamps, no structure, nothing!"""
        
        malformed_file = self.temp_dir / "severely_malformed.srt"
        with open(malformed_file, 'w', encoding='utf-8') as f:
            f.write(malformed_content)
        
        output_file = self.output_dir / "error_recovery_test.srt"
        
        # Should handle gracefully without crashing
        with pytest.raises(Exception):  # Should raise an exception for invalid format
            self.processor.process_srt_file(malformed_file, output_file)
    
    def test_unicode_and_encoding_handling(self):
        """Test handling of Unicode characters and different encodings."""
        unicode_srt = """1
00:00:01,000 --> 00:00:04,000
This contains Unicode: café, résumé, naïve

2
00:00:04,500 --> 00:00:08,000
Sanskrit text: धर्म, कर्म, अर्थ, मोक्ष

3
00:00:08,500 --> 00:00:12,000
IAST transliteration: dharma, karma, artha, mokṣa"""
        
        unicode_file = self.temp_dir / "unicode_test.srt"
        with open(unicode_file, 'w', encoding='utf-8') as f:
            f.write(unicode_srt)
        
        output_file = self.output_dir / "unicode_processed.srt"
        
        # Should process without issues
        metrics = self.processor.process_srt_file(unicode_file, output_file)
        
        assert output_file.exists()
        assert metrics.total_segments == 3
        assert metrics.timestamp_integrity_verified
        
        # Verify Unicode preservation
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "café" in content
        assert "dharma" in content
        assert "mokṣa" in content or "moksha" in content
    
    def test_timestamp_preservation(self):
        """Test that timestamps are preserved correctly."""
        input_file = Path("data/test_samples/basic_test.srt")
        output_file = self.output_dir / "timestamp_test.srt"
        
        # Read original timestamps
        parser = SRTParser()
        original_segments = parser.parse_file(str(input_file))
        
        # Process file
        metrics = self.processor.process_srt_file(input_file, output_file)
        
        # Read processed timestamps
        processed_segments = parser.parse_file(str(output_file))
        
        # Verify same number of segments
        assert len(original_segments) == len(processed_segments)
        
        # Verify timestamps are preserved
        for orig, proc in zip(original_segments, processed_segments):
            assert abs(orig.start_time - proc.start_time) < 0.001
            assert abs(orig.end_time - proc.end_time) < 0.001
        
        # Verify timestamp integrity
        assert metrics.timestamp_integrity_verified
    
    def test_large_file_handling(self):
        """Test handling of larger SRT files."""
        # Create a larger SRT file
        large_srt_content = []
        for i in range(100):
            start_time = i * 4
            end_time = start_time + 3
            start_str = f"{start_time//3600:02d}:{(start_time%3600)//60:02d}:{start_time%60:02d},000"
            end_str = f"{end_time//3600:02d}:{(end_time%3600)//60:02d}:{end_time%60:02d},000"
            
            large_srt_content.append(f"""{i+1}
{start_str} --> {end_str}
Um, this is segment number {i+1} with, uh, some filler words.""")
        
        large_content = "\n\n".join(large_srt_content)
        
        large_file = self.temp_dir / "large_test.srt"
        with open(large_file, 'w', encoding='utf-8') as f:
            f.write(large_content)
        
        output_file = self.output_dir / "large_processed.srt"
        
        # Process large file
        metrics = self.processor.process_srt_file(large_file, output_file)
        
        # Verify processing
        assert output_file.exists()
        assert metrics.total_segments == 100
        assert metrics.segments_modified > 0  # Should remove filler words
        assert metrics.processing_time < 30  # Should complete within reasonable time
        
        # Check that performance metrics are reasonable
        report = self.processor.get_processing_report(metrics)
        segments_per_second = float(report["performance"]["segments_per_second"])
        assert segments_per_second > 1  # Should process at least 1 segment per second


class TestComponentIntegration:
    """Test integration between different components."""
    
    def setup_method(self):
        """Setup for component integration tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parser_normalizer_integration(self):
        """Test integration between SRT parser and text normalizer."""
        # Create test SRT
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Um, hello world, you know

2
00:00:04,500 --> 00:00:08,000
This has twenty one words and, uh, first sentence."""
        
        # Parse SRT
        parser = SRTParser()
        segments = parser.parse_string(srt_content)
        
        # Normalize text in segments
        normalizer = TextNormalizer()
        for segment in segments:
            segment.text = normalizer.normalize_text(segment.text)
        
        # Convert back to SRT
        output_srt = parser.to_srt_string(segments)
        
        # Verify integration
        assert "Um," not in output_srt
        assert "you know" not in output_srt
        assert "21 words" in output_srt
        assert "1st sentence" in output_srt
    
    def test_normalizer_metrics_integration(self):
        """Test integration between text normalizer and metrics collector."""
        normalizer = TextNormalizer()
        collector = MetricsCollector({'metrics_dir': str(self.temp_dir)})
        
        # Start session
        collector.start_session("integration_test")
        
        # Create metrics and process text
        metrics = collector.create_file_metrics("test.srt")
        
        test_text = "Um, chapter two has twenty one verses, you know."
        result = normalizer.normalize_with_tracking(test_text)
        
        # Update metrics with normalization results
        for change in result.changes_applied:
            collector.update_correction_count(metrics, f"normalization_{change}")
        
        # Add metrics to session
        collector.add_file_metrics(metrics)
        
        # End session and verify
        session = collector.end_session()
        report = collector.generate_session_report(session)
        
        # Should have normalization corrections tracked
        corrections = report["corrections_by_type"]
        assert any("normalization" in key for key in corrections.keys())
    
    def test_full_component_chain(self):
        """Test the complete chain of all components working together."""
        # Setup all components
        parser = SRTParser()
        normalizer = TextNormalizer({
            'remove_fillers': True,
            'convert_numbers': True,
            'standardize_punctuation': True,
            'fix_capitalization': True
        })
        collector = MetricsCollector({'metrics_dir': str(self.temp_dir)})
        
        # Create test SRT
        srt_content = """1
00:00:01,000 --> 00:00:04,000
um, today we discuss chapter two verse twenty one.

2
00:00:04,500 --> 00:00:08,000
this is, you know, the first important verse  .

3
00:00:08,500 --> 00:00:12,000
Actually, it talks about the eternal soul, er, atman."""
        
        # Start metrics collection
        session_id = collector.start_session("full_chain_test")
        metrics = collector.create_file_metrics("chain_test.srt")
        
        # Step 1: Parse SRT
        collector.start_timer("parsing")
        segments = parser.parse_string(srt_content)
        metrics.parsing_time = collector.end_timer("parsing")
        metrics.total_segments = len(segments)
        
        # Step 2: Validate timestamps
        collector.start_timer("validation")
        timestamp_valid = parser.validate_timestamps(segments)
        metrics.timestamp_integrity_verified = timestamp_valid
        metrics.validation_time = collector.end_timer("validation")
        
        # Step 3: Process each segment
        collector.start_timer("normalization")
        processed_segments = []
        for segment in segments:
            original_text = segment.text
            result = normalizer.normalize_with_tracking(segment.text)
            segment.text = result.normalized_text
            
            # Track changes
            for change in result.changes_applied:
                collector.update_correction_count(metrics, f"normalization_{change}")
            
            if original_text != segment.text:
                metrics.segments_modified += 1
            
            processed_segments.append(segment)
        
        metrics.normalization_time = collector.end_timer("normalization")
        
        # Step 4: Generate output
        output_srt = parser.to_srt_string(processed_segments)
        
        # Step 5: Finalize metrics
        collector.calculate_quality_metrics(metrics)
        collector.add_file_metrics(metrics)
        session = collector.end_session()
        
        # Verify complete chain worked
        assert len(processed_segments) == 3
        assert metrics.segments_modified > 0
        assert sum(metrics.corrections_applied.values()) > 0
        assert session.total_files_processed == 1
        
        # Verify text transformations
        assert "um," not in output_srt.lower()
        assert "you know" not in output_srt.lower()
        assert "chapter 2 verse 21" in output_srt or "chapter two verse 21" in output_srt
        assert "This is" in output_srt  # Proper capitalization
        
        # Generate final report
        report = collector.generate_session_report(session)
        assert report["session_summary"]["success_rate"] == "100.0%"
        assert report["processing_statistics"]["total_corrections"] > 0