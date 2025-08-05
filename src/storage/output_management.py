"""
Output Storage Management System for ASR Post-Processing Pipeline
Story 1.1: File Naming Conventions & Storage Management

This module handles output directory mirroring, processed file naming, and metadata tracking.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml


@dataclass
class ProcessingMetadata:
    """Metadata for processed files."""
    original_file: str
    processed_file: str
    processing_version: str
    processing_timestamp: datetime
    original_size_bytes: int
    processed_size_bytes: int
    processing_duration_seconds: float
    quality_metrics: Dict
    errors: List[str]
    success: bool


class OutputStorageManager:
    """Manages output storage with directory mirroring and file versioning."""
    
    def __init__(self, config_path: str = "config/file_naming.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.base_path = Path(".")
        self.input_path = self.base_path / self.config['directories']['input']
        self.output_path = self.base_path / self.config['directories']['output']
        self.logs_path = self.base_path / "logs"
        
        # Ensure output directories exist
        self._ensure_directories()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [self.output_path, self.logs_path]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def mirror_directory_structure(self, source_file_path: Path) -> Path:
        """
        Mirror the directory structure from input to output directory.
        
        Args:
            source_file_path: Path to the source file in the input directory
            
        Returns:
            Path where the processed file should be stored
        """
        # Get relative path from input directory
        try:
            relative_path = source_file_path.relative_to(self.input_path)
        except ValueError:
            raise ValueError(f"Source file {source_file_path} is not within input directory {self.input_path}")
        
        # Create corresponding directory structure in output
        output_file_path = self.output_path / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        return output_file_path
    
    def generate_processed_filename(self, original_filename: str, version: int = 1) -> str:
        """
        Generate processed filename according to naming convention.
        
        Args:
            original_filename: Original SRT filename
            version: Processing version number
            
        Returns:
            New filename with processing suffix
        """
        # Remove extension
        name_without_ext = original_filename.replace('.srt', '')
        
        # Add processing suffix
        version_str = f"v{version}"
        processed_name = f"{name_without_ext}_processed_{version_str}.srt"
        
        return processed_name
    
    def get_next_version(self, output_file_path: Path) -> int:
        """
        Get the next available version number for a processed file.
        
        Args:
            output_file_path: Base output file path
            
        Returns:
            Next available version number
        """
        base_name = output_file_path.stem.replace('.srt', '')
        parent_dir = output_file_path.parent
        
        version = 1
        while True:
            version_filename = f"{base_name}_processed_v{version}.srt"
            if not (parent_dir / version_filename).exists():
                break
            version += 1
        
        return version
    
    def store_processed_file(self, 
                           source_file_path: Path, 
                           processed_content: str,
                           processing_metadata: Optional[Dict] = None) -> Tuple[Path, ProcessingMetadata]:
        """
        Store a processed file with proper naming and metadata tracking.
        
        Args:
            source_file_path: Path to the original source file
            processed_content: Processed file content
            processing_metadata: Optional metadata about the processing
            
        Returns:
            Tuple of (output_file_path, processing_metadata)
        """
        start_time = datetime.now()
        
        # Mirror directory structure
        base_output_path = self.mirror_directory_structure(source_file_path)
        
        # Generate processed filename with version
        version = self.get_next_version(base_output_path)
        processed_filename = self.generate_processed_filename(source_file_path.name, version)
        
        # Final output path
        output_file_path = base_output_path.parent / processed_filename
        
        # Write processed content
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            processing_success = True
            processing_errors = []
        except Exception as e:
            processing_success = False
            processing_errors = [str(e)]
        
        # Calculate processing duration
        processing_duration = (datetime.now() - start_time).total_seconds()
        
        # Create processing metadata
        metadata = ProcessingMetadata(
            original_file=str(source_file_path),
            processed_file=str(output_file_path),
            processing_version=f"v{version}",
            processing_timestamp=start_time,
            original_size_bytes=source_file_path.stat().st_size if source_file_path.exists() else 0,
            processed_size_bytes=output_file_path.stat().st_size if output_file_path.exists() else 0,
            processing_duration_seconds=processing_duration,
            quality_metrics=processing_metadata or {},
            errors=processing_errors,
            success=processing_success
        )
        
        # Store metadata
        self._store_processing_metadata(metadata)
        
        return output_file_path, metadata
    
    def _store_processing_metadata(self, metadata: ProcessingMetadata):
        """Store processing metadata to log file."""
        log_date = metadata.processing_timestamp.strftime('%Y%m%d')
        log_filename = f"processing_log_{log_date}.json"
        log_file_path = self.logs_path / log_filename
        
        # Load existing log or create new
        if log_file_path.exists():
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
        else:
            log_data = {
                'date': log_date,
                'processing_entries': []
            }
        
        # Add new entry
        metadata_dict = asdict(metadata)
        metadata_dict['processing_timestamp'] = metadata.processing_timestamp.isoformat()
        log_data['processing_entries'].append(metadata_dict)
        
        # Save updated log
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def create_processing_manifest(self, processed_files: List[ProcessingMetadata]) -> Path:
        """
        Create a processing manifest file summarizing all processed files.
        
        Args:
            processed_files: List of processing metadata
            
        Returns:
            Path to the created manifest file
        """
        timestamp = datetime.now()
        manifest_filename = f"processing_manifest_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        manifest_path = self.logs_path / manifest_filename
        
        manifest_data = {
            'creation_timestamp': timestamp.isoformat(),
            'total_files_processed': len(processed_files),
            'successful_processings': len([f for f in processed_files if f.success]),
            'failed_processings': len([f for f in processed_files if not f.success]),
            'total_processing_time_seconds': sum(f.processing_duration_seconds for f in processed_files),
            'files': [asdict(f) for f in processed_files]
        }
        
        # Convert datetime objects to ISO strings
        for file_data in manifest_data['files']:
            if 'processing_timestamp' in file_data:
                file_data['processing_timestamp'] = file_data['processing_timestamp'].isoformat()
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        
        return manifest_path
    
    def cleanup_old_versions(self, keep_versions: int = 3):
        """
        Clean up old versions of processed files, keeping only the most recent.
        
        Args:
            keep_versions: Number of versions to keep for each file
        """
        # Group files by base name
        file_groups = {}
        
        for file_path in self.output_path.rglob("*_processed_v*.srt"):
            # Extract base name (everything before _processed_v)
            base_name = file_path.name.split('_processed_v')[0]
            
            if base_name not in file_groups:
                file_groups[base_name] = []
            
            file_groups[base_name].append(file_path)
        
        # Clean up old versions for each group
        for base_name, files in file_groups.items():
            if len(files) <= keep_versions:
                continue
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old versions
            for old_file in files[keep_versions:]:
                try:
                    old_file.unlink()
                    print(f"Removed old version: {old_file}")
                except Exception as e:
                    print(f"Error removing {old_file}: {e}")
    
    def get_processing_statistics(self, days: int = 30) -> Dict:
        """
        Get processing statistics for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary containing processing statistics
        """
        stats = {
            'total_files_processed': 0,
            'successful_processings': 0,
            'failed_processings': 0,
            'total_processing_time_seconds': 0,
            'average_processing_time_seconds': 0,
            'files_by_day': {},
            'errors': []
        }
        
        # Look through log files for the specified period
        end_date = datetime.now()
        start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0) - \
                    datetime.timedelta(days=days)
        
        for log_file in self.logs_path.glob("processing_log_*.json"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                for entry in log_data.get('processing_entries', []):
                    entry_time = datetime.fromisoformat(entry['processing_timestamp'])
                    
                    if start_date <= entry_time <= end_date:
                        stats['total_files_processed'] += 1
                        
                        if entry['success']:
                            stats['successful_processings'] += 1
                        else:
                            stats['failed_processings'] += 1
                            stats['errors'].extend(entry.get('errors', []))
                        
                        stats['total_processing_time_seconds'] += entry['processing_duration_seconds']
                        
                        # Group by day
                        day_key = entry_time.strftime('%Y-%m-%d')
                        if day_key not in stats['files_by_day']:
                            stats['files_by_day'][day_key] = 0
                        stats['files_by_day'][day_key] += 1
                        
            except Exception as e:
                print(f"Error reading log file {log_file}: {e}")
        
        # Calculate average
        if stats['total_files_processed'] > 0:
            stats['average_processing_time_seconds'] = \
                stats['total_processing_time_seconds'] / stats['total_files_processed']
        
        return stats


def main():
    """Example usage of the output storage manager."""
    output_manager = OutputStorageManager()
    
    # Example: Store a processed file
    source_file = Path("data/raw_srts/VED001_20241201_SwamiBrahmananda.srt")
    processed_content = "1\n00:00:00,000 --> 00:00:05,000\nProcessed example content\n"
    
    if source_file.exists():
        try:
            output_path, metadata = output_manager.store_processed_file(
                source_file, 
                processed_content,
                {'word_error_rate': 0.05, 'confidence_score': 0.95}
            )
            print(f"Processed file stored at: {output_path}")
            print(f"Processing successful: {metadata.success}")
        except Exception as e:
            print(f"Error storing processed file: {e}")
    
    # Get processing statistics
    stats = output_manager.get_processing_statistics(days=7)
    print(f"\nProcessing Statistics (last 7 days):")
    print(f"Total files processed: {stats['total_files_processed']}")
    print(f"Success rate: {stats['successful_processings']}/{stats['total_files_processed']}")


if __name__ == "__main__":
    main()