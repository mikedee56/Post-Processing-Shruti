"""
Storage Management Module
Handles file ingestion, output management, and directory operations.
"""

from .file_ingestion import FileIngestionSystem, FileInfo
from .output_management import OutputStorageManager, ProcessingMetadata

__all__ = [
    'FileIngestionSystem',
    'FileInfo', 
    'OutputStorageManager',
    'ProcessingMetadata'
]