"""
File utility functions for ASR post-processing workflow.

This module provides common file handling operations including
SRT file validation, encoding detection, and file system utilities.
"""

import os
import re
import chardet
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging

import pysrt
import pandas as pd


logger = logging.getLogger(__name__)


def detect_file_encoding(file_path: Path) -> str:
    """
    Detect the encoding of a text file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding string
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        logger.debug(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence:.2f})")
        
        # Default to UTF-8 if detection is uncertain
        if confidence < 0.7:
            logger.warning(f"Low confidence encoding detection for {file_path}, defaulting to UTF-8")
            encoding = 'utf-8'
            
        return encoding
        
    except Exception as e:
        logger.error(f"Error detecting encoding for {file_path}: {e}")
        return 'utf-8'  # Default fallback


def validate_srt_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate an SRT file for format correctness and basic structure.
    
    Args:
        file_path: Path to SRT file
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': False,
        'errors': [],
        'warnings': [],
        'subtitle_count': 0,
        'encoding': 'unknown',
        'file_size': 0,
        'duration': 0.0
    }
    
    try:
        # Check file exists and is readable
        if not file_path.exists():
            validation_result['errors'].append(f"File does not exist: {file_path}")
            return validation_result
        
        # Get file stats
        stat = file_path.stat()
        validation_result['file_size'] = stat.st_size
        
        if validation_result['file_size'] == 0:
            validation_result['errors'].append("File is empty")
            return validation_result
        
        # Detect encoding
        validation_result['encoding'] = detect_file_encoding(file_path)
        
        # Try to parse as SRT
        try:
            subs = pysrt.open(file_path, encoding=validation_result['encoding'])
            validation_result['subtitle_count'] = len(subs)
            
            if len(subs) == 0:
                validation_result['errors'].append("No subtitles found in file")
                return validation_result
            
            # Calculate duration
            if subs:
                last_sub = subs[-1]
                validation_result['duration'] = _time_to_seconds(last_sub.end)
            
            # Validate subtitle structure
            for i, sub in enumerate(subs):
                # Check for required components
                if not sub.text or not sub.text.strip():
                    validation_result['warnings'].append(f"Empty text in subtitle {i+1}")
                
                # Check time validity
                if sub.start >= sub.end:
                    validation_result['errors'].append(f"Invalid time range in subtitle {i+1}")
                
                # Check for overlapping subtitles
                if i > 0 and sub.start < subs[i-1].end:
                    validation_result['warnings'].append(f"Overlapping subtitles at {i+1}")
            
            validation_result['is_valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['errors'].append(f"SRT parsing error: {e}")
    
    except Exception as e:
        validation_result['errors'].append(f"File validation error: {e}")
    
    return validation_result


def normalize_path(path: str) -> Path:
    """
    Normalize a file path for cross-platform compatibility.
    
    Args:
        path: String path to normalize
        
    Returns:
        Normalized Path object
    """
    # Convert to Path object
    normalized = Path(path)
    
    # Resolve relative paths
    if not normalized.is_absolute():
        normalized = Path.cwd() / normalized
    
    # Resolve any .. or . components
    normalized = normalized.resolve()
    
    return normalized


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing or replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    # Remove or replace problematic characters
    safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    safe_chars = safe_chars.strip('. ')
    
    # Limit length
    if len(safe_chars) > 255:
        name, ext = os.path.splitext(safe_chars)
        safe_chars = name[:255-len(ext)] + ext
    
    # Ensure not empty
    if not safe_chars:
        safe_chars = "untitled"
    
    return safe_chars


def create_backup(file_path: Path, backup_dir: Optional[Path] = None) -> Path:
    """
    Create a backup copy of a file.
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup (optional)
        
    Returns:
        Path to backup file
    """
    if backup_dir is None:
        backup_dir = file_path.parent / 'backups'
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique backup filename
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    # Copy file
    import shutil
    shutil.copy2(file_path, backup_path)
    
    logger.info(f"Created backup: {backup_path}")
    return backup_path


def find_files_by_pattern(directory: Path, pattern: str, recursive: bool = True) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    if recursive:
        matches = list(directory.rglob(pattern))
    else:
        matches = list(directory.glob(pattern))
    
    # Filter to only files (not directories)
    file_matches = [path for path in matches if path.is_file()]
    
    logger.debug(f"Found {len(file_matches)} files matching '{pattern}' in {directory}")
    
    return file_matches


def get_file_stats(file_path: Path) -> Dict[str, Any]:
    """
    Get comprehensive statistics about a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file statistics
    """
    try:
        stat = file_path.stat()
        
        stats = {
            'path': str(file_path),
            'name': file_path.name,
            'size_bytes': stat.st_size,
            'size_human': _format_file_size(stat.st_size),
            'created': pd.Timestamp.fromtimestamp(stat.st_ctime),
            'modified': pd.Timestamp.fromtimestamp(stat.st_mtime),
            'accessed': pd.Timestamp.fromtimestamp(stat.st_atime),
            'is_readable': os.access(file_path, os.R_OK),
            'is_writable': os.access(file_path, os.W_OK),
        }
        
        # Add SRT-specific stats if it's an SRT file
        if file_path.suffix.lower() == '.srt':
            srt_validation = validate_srt_file(file_path)
            stats.update({
                'subtitle_count': srt_validation['subtitle_count'],
                'duration': srt_validation['duration'],
                'srt_valid': srt_validation['is_valid'],
                'encoding': srt_validation['encoding']
            })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting file stats for {file_path}: {e}")
        return {'path': str(file_path), 'error': str(e)}


def _time_to_seconds(time_obj) -> float:
    """Convert pysrt time object to seconds."""
    return time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds + time_obj.milliseconds / 1000.0


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def ensure_directory_exists(directory: Path, create_parents: bool = True) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        create_parents: Whether to create parent directories
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        if not directory.exists():
            directory.mkdir(parents=create_parents, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        return directory.exists() and directory.is_dir()
        
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False


def clean_filename_for_processing(filename: str) -> str:
    """
    Clean a filename for processing while preserving important information.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove file extension for processing
    name_without_ext = Path(filename).stem
    
    # Replace common separators with underscores
    cleaned = re.sub(r'[-\s\.]+', '_', name_without_ext)
    
    # Remove special characters but keep alphanumeric and underscores
    cleaned = re.sub(r'[^\w_]', '', cleaned)
    
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_{2,}', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    # Ensure not empty
    if not cleaned:
        cleaned = "processed_file"
    
    return cleaned