"""
Unit tests for SRT Parser functionality.

Tests comprehensive SRT parsing, validation, and error handling
with various SRT format variations and edge cases.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List

from src.utils.srt_parser import SRTParser, SRTSegment, SRTParseError


class TestSRTParser:
    """Test suite for SRTParser class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.parser = SRTParser()
    
    def test_parse_valid_srt_string(self):
        """Test parsing a valid SRT string."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:04,500 --> 00:00:08,000
This is a test."""
        
        segments = self.parser.parse_string(srt_content)
        
        assert len(segments) == 2
        assert segments[0].index == 1
        assert segments[0].start_time == 1.0
        assert segments[0].end_time == 4.0
        assert segments[0].text == "Hello world"
        
        assert segments[1].index == 2
        assert segments[1].start_time == 4.5
        assert segments[1].end_time == 8.0
        assert segments[1].text == "This is a test."
    
    def test_parse_empty_content(self):
        """Test parsing empty content."""
        segments = self.parser.parse_string("")
        assert len(segments) == 0
        
        segments = self.parser.parse_string("   \n\n  ")
        assert len(segments) == 0
    
    def test_parse_invalid_timestamp_format(self):
        """Test parsing with invalid timestamp format."""
        srt_content = """1
invalid timestamp format
Hello world"""
        
        segments = self.parser.parse_string(srt_content)
        assert len(segments) == 0  # Should skip invalid blocks
    
    def test_parse_missing_text(self):
        """Test parsing with missing text content."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000"""
        
        segments = self.parser.parse_string(srt_content)
        assert len(segments) == 0  # Should skip incomplete blocks
    
    def test_parse_multiline_text(self):
        """Test parsing segments with multiline text."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
First line
Second line
Third line"""
        
        segments = self.parser.parse_string(srt_content)
        
        assert len(segments) == 1
        assert segments[0].text == "First line\nSecond line\nThird line"
    
    def test_parse_with_html_tags(self):
        """Test parsing segments with HTML tags."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
<i>Italic text</i> and <b>bold text</b>"""
        
        segments = self.parser.parse_string(srt_content)
        
        assert len(segments) == 1
        assert segments[0].text == "Italic text and bold text"  # HTML tags removed
    
    def test_parse_with_different_separators(self):
        """Test parsing with different time separators (comma vs period)."""
        srt_content_comma = """1
00:00:01,000 --> 00:00:04,000
Text with comma separator"""
        
        srt_content_period = """1
00:00:01.000 --> 00:00:04.000
Text with period separator"""
        
        segments_comma = self.parser.parse_string(srt_content_comma)
        segments_period = self.parser.parse_string(srt_content_period)
        
        assert len(segments_comma) == 1
        assert len(segments_period) == 1
        assert segments_comma[0].start_time == segments_period[0].start_time
    
    def test_parse_with_irregular_spacing(self):
        """Test parsing with irregular spacing and line endings."""
        srt_content = """1


00:00:01,000   -->   00:00:04,000


Hello world


2
00:00:04,500 --> 00:00:08,000
This is a test.


"""
        
        segments = self.parser.parse_string(srt_content)
        
        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[1].text == "This is a test."
    
    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_file("non_existent_file.srt")
    
    def test_parse_file_success(self):
        """Test successful file parsing."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello from file

2
00:00:04,500 --> 00:00:08,000
File parsing test."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(srt_content)
            temp_path = f.name
        
        try:
            segments = self.parser.parse_file(temp_path)
            
            assert len(segments) == 2
            assert segments[0].text == "Hello from file"
            assert segments[1].text == "File parsing test."
        finally:
            Path(temp_path).unlink()
    
    def test_validate_timestamps_valid(self):
        """Test timestamp validation with valid timestamps."""
        segments = [
            SRTSegment(1, 1.0, 4.0, "First", "First"),
            SRTSegment(2, 4.5, 8.0, "Second", "Second"),
            SRTSegment(3, 8.5, 12.0, "Third", "Third")
        ]
        
        assert self.parser.validate_timestamps(segments) is True
    
    def test_validate_timestamps_invalid_individual(self):
        """Test timestamp validation with invalid individual timing."""
        segments = [
            SRTSegment(1, 4.0, 1.0, "Invalid", "Invalid")  # start > end
        ]
        
        assert self.parser.validate_timestamps(segments) is False
    
    def test_validate_timestamps_negative(self):
        """Test timestamp validation with negative timestamps."""
        segments = [
            SRTSegment(1, -1.0, 4.0, "Negative start", "Negative start")
        ]
        
        assert self.parser.validate_timestamps(segments) is False
    
    def test_validate_timestamps_overlapping(self):
        """Test timestamp validation with overlapping segments."""
        segments = [
            SRTSegment(1, 1.0, 5.0, "First", "First"),
            SRTSegment(2, 3.0, 8.0, "Overlapping", "Overlapping")  # Overlaps with first
        ]
        
        # Overlapping is a warning, not an error
        assert self.parser.validate_timestamps(segments) is True
    
    def test_validate_timestamps_empty(self):
        """Test timestamp validation with empty list."""
        assert self.parser.validate_timestamps([]) is True
    
    def test_to_srt_string(self):
        """Test converting segments back to SRT string."""
        segments = [
            SRTSegment(1, 1.0, 4.0, "First segment", "First segment"),
            SRTSegment(2, 4.5, 8.0, "Second segment", "Second segment")
        ]
        
        srt_string = self.parser.to_srt_string(segments)
        
        expected = """1
00:00:01,000 --> 00:00:04,000
First segment

2
00:00:04,500 --> 00:00:08,000
Second segment"""
        
        assert srt_string == expected
    
    def test_to_srt_string_empty(self):
        """Test converting empty segments to SRT string."""
        srt_string = self.parser.to_srt_string([])
        assert srt_string == ""
    
    def test_round_trip_parsing(self):
        """Test that parsing and converting back produces equivalent results."""
        original_srt = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:04,500 --> 00:00:08,000
This is a test."""
        
        # Parse and convert back
        segments = self.parser.parse_string(original_srt)
        converted_srt = self.parser.to_srt_string(segments)
        
        # Parse the converted SRT
        segments_2 = self.parser.parse_string(converted_srt)
        
        # Should have same number of segments
        assert len(segments) == len(segments_2)
        
        # Content should match
        for i in range(len(segments)):
            assert segments[i].index == segments_2[i].index
            assert abs(segments[i].start_time - segments_2[i].start_time) < 0.001
            assert abs(segments[i].end_time - segments_2[i].end_time) < 0.001
            assert segments[i].text == segments_2[i].text
    
    def test_time_conversion_edge_cases(self):
        """Test time conversion with edge cases."""
        # Test hours, minutes, seconds, milliseconds
        srt_content = """1
01:23:45,678 --> 02:34:56,789
Long timestamp test"""
        
        segments = self.parser.parse_string(srt_content)
        
        assert len(segments) == 1
        # 1*3600 + 23*60 + 45 + 0.678 = 5025.678
        assert abs(segments[0].start_time - 5025.678) < 0.001
        # 2*3600 + 34*60 + 56 + 0.789 = 9296.789
        assert abs(segments[0].end_time - 9296.789) < 0.001
    
    def test_malformed_index_handling(self):
        """Test handling of malformed segment indices."""
        srt_content = """invalid_index
00:00:01,000 --> 00:00:04,000
Text with invalid index

2
00:00:04,500 --> 00:00:08,000
Valid segment"""
        
        segments = self.parser.parse_string(srt_content)
        
        # Should still parse the valid segment and handle invalid index gracefully
        assert len(segments) >= 1
        # Should use expected index for malformed entries
        if len(segments) == 2:
            assert segments[0].index == 1  # Corrected from invalid
        assert segments[-1].index == 2
    
    def test_encoding_handling(self):
        """Test handling different file encodings."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Text with special characters: café résumé naïve"""
        
        # Test UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(srt_content)
            temp_path = f.name
        
        try:
            segments = self.parser.parse_file(temp_path)
            assert len(segments) == 1
            assert "café" in segments[0].text
        finally:
            Path(temp_path).unlink()


class TestSRTSegment:
    """Test suite for SRTSegment dataclass."""
    
    def test_segment_creation(self):
        """Test creating SRTSegment instances."""
        segment = SRTSegment(
            index=1,
            start_time=1.0,
            end_time=4.0,
            text="Test text",
            raw_text="Test text"
        )
        
        assert segment.index == 1
        assert segment.start_time == 1.0
        assert segment.end_time == 4.0
        assert segment.text == "Test text"
        assert segment.raw_text == "Test text"
        assert segment.confidence is None
        assert segment.processing_flags == []
    
    def test_segment_with_optional_fields(self):
        """Test creating SRTSegment with optional fields."""
        segment = SRTSegment(
            index=1,
            start_time=1.0,
            end_time=4.0,
            text="Test text",
            raw_text="Test text",
            confidence=0.85,
            processing_flags=["test_flag"]
        )
        
        assert segment.confidence == 0.85
        assert segment.processing_flags == ["test_flag"]


class TestSRTParseError:
    """Test suite for SRTParseError exception."""
    
    def test_parse_error_creation(self):
        """Test creating SRTParseError."""
        error = SRTParseError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)