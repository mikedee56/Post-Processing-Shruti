"""
File Ingestion System for ASR Post-Processing Pipeline
Story 1.1: File Naming Conventions & Storage Management

This module handles file discovery, validation, and ingestion from the raw_srts directory.
"""

import os
import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FileInfo:
    """Information about a discovered file."""
    filepath: Path
    filename: str
    lecture_id: str
    date: str
    speaker: str
    size_bytes: int
    created_time: datetime
    is_valid: bool
    validation_errors: List[str]


class FileIngestionSystem:
    """Handles file discovery and validation for SRT files."""
    
    def __init__(self, config_path: str = "config/file_naming.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.base_path = Path(".")
        self.raw_srts_path = self.base_path / self.config['directories']['input']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def discover_files(self) -> List[FileInfo]:
        """Discover all SRT files in the raw_srts directory."""
        files = []
        
        if not self.raw_srts_path.exists():
            raise FileNotFoundError(f"Raw SRTs directory not found: {self.raw_srts_path}")
        
        for filepath in self.raw_srts_path.rglob("*.srt"):
            file_info = self._analyze_file(filepath)
            files.append(file_info)
        
        return files
    
    def _analyze_file(self, filepath: Path) -> FileInfo:
        """Analyze a single file and extract metadata."""
        filename = filepath.name
        size_bytes = filepath.stat().st_size
        created_time = datetime.fromtimestamp(filepath.stat().st_ctime)
        
        # Parse filename according to naming convention
        lecture_id, date, speaker, is_valid, errors = self._parse_filename(filename)
        
        # Additional validation
        additional_errors = self._validate_file(filepath)
        errors.extend(additional_errors)
        is_valid = is_valid and len(additional_errors) == 0
        
        return FileInfo(
            filepath=filepath,
            filename=filename,
            lecture_id=lecture_id,
            date=date,
            speaker=speaker,
            size_bytes=size_bytes,
            created_time=created_time,
            is_valid=is_valid,
            validation_errors=errors
        )
    
    def _parse_filename(self, filename: str) -> Tuple[str, str, str, bool, List[str]]:
        """Parse filename according to naming convention."""
        errors = []
        
        # Remove extension
        name_without_ext = filename.replace('.srt', '')
        
        # Expected pattern: {lecture_id}_{date}_{speaker}
        pattern = r'^(VED\d{3})_(\d{8})_([A-Za-z]+)$'
        match = re.match(pattern, name_without_ext)
        
        if not match:
            errors.append(f"Filename doesn't match expected pattern: {self.config['file_naming']['srt_files']['pattern']}")
            return "", "", "", False, errors
        
        lecture_id, date, speaker = match.groups()
        
        # Validate lecture_id format
        if not re.match(r'^VED\d{3}$', lecture_id):
            errors.append(f"Invalid lecture_id format: {lecture_id}")
        
        # Validate date format
        try:
            datetime.strptime(date, '%Y%m%d')
        except ValueError:
            errors.append(f"Invalid date format: {date}")
        
        # Validate speaker format (CamelCase)
        if not re.match(r'^[A-Z][a-zA-Z]*$', speaker):
            errors.append(f"Invalid speaker format (should be CamelCase): {speaker}")
        
        return lecture_id, date, speaker, len(errors) == 0, errors
    
    def _validate_file(self, filepath: Path) -> List[str]:
        """Perform additional file validation."""
        errors = []
        
        # Check file size
        max_size_mb = self.config['validation']['srt_files']['max_size_mb']
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            errors.append(f"File size ({size_mb:.2f}MB) exceeds maximum ({max_size_mb}MB)")
        
        # Check file extension
        required_extensions = self.config['validation']['srt_files']['required_extensions']
        if filepath.suffix not in required_extensions:
            errors.append(f"Invalid file extension: {filepath.suffix}")
        
        # Check filename length
        max_length = self.config['validation']['naming_constraints']['max_filename_length']
        if len(filepath.name) > max_length:
            errors.append(f"Filename too long: {len(filepath.name)} > {max_length}")
        
        # Check forbidden characters
        forbidden_chars = self.config['validation']['naming_constraints']['forbidden_chars']
        for char in forbidden_chars:
            if char in filepath.name:
                errors.append(f"Forbidden character in filename: {char}")
        
        # Check file encoding (basic check)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                f.read(100)  # Read first 100 chars to test encoding
        except UnicodeDecodeError:
            errors.append("File is not UTF-8 encoded")
        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")
        
        return errors
    
    def get_valid_files(self) -> List[FileInfo]:
        """Get only valid files that pass all validation checks."""
        all_files = self.discover_files()
        return [f for f in all_files if f.is_valid]
    
    def get_invalid_files(self) -> List[FileInfo]:
        """Get files that failed validation."""
        all_files = self.discover_files()
        return [f for f in all_files if not f.is_valid]
    
    def generate_ingestion_report(self) -> Dict:
        """Generate a comprehensive ingestion report."""
        all_files = self.discover_files()
        valid_files = [f for f in all_files if f.is_valid]
        invalid_files = [f for f in all_files if not f.is_valid]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'directory_scanned': str(self.raw_srts_path),
            'total_files_found': len(all_files),
            'valid_files': len(valid_files),
            'invalid_files': len(invalid_files),
            'validation_success_rate': len(valid_files) / len(all_files) * 100 if all_files else 0,
            'files': {
                'valid': [
                    {
                        'filename': f.filename,
                        'lecture_id': f.lecture_id,
                        'date': f.date,
                        'speaker': f.speaker,
                        'size_mb': f.size_bytes / (1024 * 1024)
                    }
                    for f in valid_files
                ],
                'invalid': [
                    {
                        'filename': f.filename,
                        'errors': f.validation_errors,
                        'size_mb': f.size_bytes / (1024 * 1024)
                    }
                    for f in invalid_files
                ]
            }
        }
        
        return report


def main():
    """Example usage of the file ingestion system."""
    ingestion = FileIngestionSystem()
    
    try:
        report = ingestion.generate_ingestion_report()
        print(f"File Ingestion Report:")
        print(f"Total files: {report['total_files_found']}")
        print(f"Valid files: {report['valid_files']}")
        print(f"Invalid files: {report['invalid_files']}")
        print(f"Success rate: {report['validation_success_rate']:.1f}%")
        
        if report['invalid_files'] > 0:
            print("\nInvalid files found:")
            for file_info in report['files']['invalid']:
                print(f"  - {file_info['filename']}: {', '.join(file_info['errors'])}")
                
    except Exception as e:
        print(f"Error during file ingestion: {e}")


if __name__ == "__main__":
    main()