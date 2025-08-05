"""
SRT Parser for processing subtitle files with timestamp preservation.

This module provides robust SRT file parsing functionality with comprehensive
error handling, validation, and support for various SRT format variations.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class SRTSegment:
    """Represents a single SRT subtitle segment with timing and text."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    raw_text: str      # original before processing
    confidence: Optional[float] = None
    processing_flags: List[str] = field(default_factory=list)


class SRTParseError(Exception):
    """Custom exception for SRT parsing errors."""
    pass


class SRTParser:
    """
    Robust SRT file parser with comprehensive format support and error handling.
    
    Handles various SRT format variations, malformed entries, and provides
    detailed validation and error reporting.
    """
    
    def __init__(self):
        """Initialize the SRT parser."""
        self.logger = logging.getLogger(__name__)
        
        # SRT timestamp pattern - supports various formats
        self.timestamp_pattern = re.compile(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})'
        )
        
        # Pattern for segment index
        self.index_pattern = re.compile(r'^\d+$')
    
    def parse_file(self, file_path: str) -> List[SRTSegment]:
        """
        Parse an SRT file and return a list of segments.
        
        Args:
            file_path: Path to the SRT file
            
        Returns:
            List of SRTSegment objects
            
        Raises:
            SRTParseError: If file cannot be parsed
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"SRT file not found: {file_path}")
        
        try:
            # Try UTF-8 first, then fall back to other encodings
            content = self._read_file_with_encoding(file_path)
            return self.parse_string(content)
            
        except Exception as e:
            self.logger.error(f"Failed to parse SRT file {file_path}: {e}")
            raise SRTParseError(f"Error parsing SRT file: {e}")
    
    def parse_string(self, srt_content: str) -> List[SRTSegment]:
        """
        Parse SRT content from a string.
        
        Args:
            srt_content: Raw SRT content as string
            
        Returns:
            List of SRTSegment objects
            
        Raises:
            SRTParseError: If content cannot be parsed
        """
        if not srt_content.strip():
            self.logger.warning("Empty SRT content provided")
            return []
        
        segments = []
        blocks = self._split_into_blocks(srt_content)
        
        for i, block in enumerate(blocks):
            try:
                segment = self._parse_block(block, i + 1)
                if segment:
                    segments.append(segment)
            except Exception as e:
                self.logger.warning(f"Failed to parse block {i + 1}: {e}")
                # Continue processing other blocks
                continue
        
        if not segments:
            raise SRTParseError("No valid SRT segments found")
        
        # Validate segments
        self._validate_segments(segments)
        
        return segments
    
    def validate_timestamps(self, segments: List[SRTSegment]) -> bool:
        """
        Validate timestamp integrity of segments.
        
        Args:
            segments: List of SRT segments to validate
            
        Returns:
            True if timestamps are valid, False otherwise
        """
        if not segments:
            return True
        
        try:
            for i, segment in enumerate(segments):
                # Check individual segment timing
                if segment.start_time >= segment.end_time:
                    self.logger.error(f"Segment {segment.index}: start time >= end time")
                    return False
                
                # Check for negative times
                if segment.start_time < 0 or segment.end_time < 0:
                    self.logger.error(f"Segment {segment.index}: negative timestamp")
                    return False
                
                # Check chronological order with next segment
                if i < len(segments) - 1:
                    next_segment = segments[i + 1]
                    if segment.end_time > next_segment.start_time:
                        self.logger.warning(f"Segments {segment.index} and {next_segment.index}: overlapping timestamps")
                        # This is a warning, not an error - overlaps can be valid
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating timestamps: {e}")
            return False
    
    def to_srt_string(self, segments: List[SRTSegment]) -> str:
        """
        Convert segments back to SRT format string.
        
        Args:
            segments: List of SRT segments
            
        Returns:
            SRT formatted string
        """
        if not segments:
            return ""
        
        srt_blocks = []
        
        for segment in segments:
            # Format timestamps
            start_time = self._seconds_to_srt_time(segment.start_time)
            end_time = self._seconds_to_srt_time(segment.end_time)
            
            # Build SRT block
            block = f"{segment.index}\n{start_time} --> {end_time}\n{segment.text}\n"
            srt_blocks.append(block)
        
        return "\n".join(srt_blocks)
    
    def _read_file_with_encoding(self, file_path: Path) -> str:
        """Try to read file with different encodings."""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                self.logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with error handling
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            self.logger.warning(f"Read {file_path} with UTF-8 and error replacement")
            return content
        except Exception as e:
            raise SRTParseError(f"Could not read file with any encoding: {e}")
    
    def _split_into_blocks(self, content: str) -> List[str]:
        """Split SRT content into individual subtitle blocks."""
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split by double newlines (empty lines separate blocks)
        blocks = re.split(r'\n\s*\n', content.strip())
        
        # Filter out empty blocks
        return [block.strip() for block in blocks if block.strip()]
    
    def _parse_block(self, block: str, expected_index: int) -> Optional[SRTSegment]:
        """Parse a single SRT block into a segment."""
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if len(lines) < 3:
            self.logger.warning(f"Block {expected_index}: insufficient lines ({len(lines)})")
            return None
        
        # Parse index
        index_line = lines[0]
        if not self.index_pattern.match(index_line):
            self.logger.warning(f"Block {expected_index}: invalid index format: {index_line}")
            # Use expected index as fallback
            index = expected_index
        else:
            index = int(index_line)
        
        # Parse timestamps
        timestamp_line = lines[1]
        timestamp_match = self.timestamp_pattern.match(timestamp_line)
        
        if not timestamp_match:
            self.logger.warning(f"Block {index}: invalid timestamp format: {timestamp_line}")
            return None
        
        # Extract timing
        groups = timestamp_match.groups()
        start_time = self._time_components_to_seconds(
            int(groups[0]), int(groups[1]), int(groups[2]), int(groups[3])
        )
        end_time = self._time_components_to_seconds(
            int(groups[4]), int(groups[5]), int(groups[6]), int(groups[7])
        )
        
        # Combine text lines
        text_lines = lines[2:]
        text = '\n'.join(text_lines)
        
        # Clean up text
        cleaned_text = self._clean_text(text)
        
        return SRTSegment(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=cleaned_text,
            raw_text=text
        )
    
    def _time_components_to_seconds(self, hours: int, minutes: int, seconds: int, milliseconds: int) -> float:
        """Convert time components to total seconds."""
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    
    def _seconds_to_srt_time(self, total_seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def _clean_text(self, text: str) -> str:
        """Clean up subtitle text."""
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _validate_segments(self, segments: List[SRTSegment]) -> None:
        """Validate parsed segments for common issues."""
        if not segments:
            return
        
        # Check for duplicate indices
        indices = [seg.index for seg in segments]
        if len(indices) != len(set(indices)):
            self.logger.warning("Duplicate segment indices found")
        
        # Check for empty text
        empty_segments = [seg.index for seg in segments if not seg.text.strip()]
        if empty_segments:
            self.logger.warning(f"Empty text segments found: {empty_segments}")
        
        # Validate timing sequence
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            if current.start_time >= current.end_time:
                self.logger.error(f"Segment {current.index}: invalid timing")
        
        self.logger.info(f"Parsed {len(segments)} segments successfully")