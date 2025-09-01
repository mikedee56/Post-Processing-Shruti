"""
Epic 4 - Story 4.1: Batch Processing Framework
BatchProcessor utility for managing large-scale SRT processing operations

This module provides batch processing capabilities with:
- Resource management and optimization
- Progress tracking and monitoring
- Error handling and recovery
- Performance metrics collection
"""

import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import threading
from queue import Queue

from .metrics_collector import MetricsCollector
from .recovery_manager import RecoveryManager


@dataclass
class BatchConfig:
    """Configuration for batch processing operations."""
    batch_size: int = 50
    max_workers: int = mp.cpu_count()
    chunk_size: int = 10
    timeout_seconds: int = 3600
    memory_limit_mb: int = 4096
    retry_attempts: int = 3
    retry_delay: float = 1.0
    progress_interval: int = 10


@dataclass
class ProcessingResult:
    """Result from processing a batch of files."""
    batch_id: str
    total_files: int
    processed_files: int
    failed_files: int
    processing_time: float
    throughput: float
    success_rate: float
    errors: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    start_time: datetime
    end_time: datetime


class ResourceMonitor:
    """Monitor system resources during batch processing."""
    
    def __init__(self, memory_limit_mb: int = 4096, check_interval: float = 5.0):
        self.memory_limit_mb = memory_limit_mb
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self._stop_event = threading.Event()
        
        # Metrics
        self.peak_memory_mb = 0
        self.peak_cpu_percent = 0
        self.memory_warnings = 0
        
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring:
            return
            
        self.monitoring = False
        self._stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()
        
        while not self._stop_event.wait(self.check_interval):
            try:
                # Check memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                
                if memory_mb > self.memory_limit_mb:
                    self.memory_warnings += 1
                    logging.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.memory_limit_mb}MB")
                
                # Check CPU usage
                cpu_percent = process.cpu_percent()
                self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
                
            except Exception as e:
                logging.warning(f"Error monitoring resources: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            return {
                'current_memory_mb': memory_mb,
                'current_cpu_percent': cpu_percent,
                'peak_memory_mb': self.peak_memory_mb,
                'peak_cpu_percent': self.peak_cpu_percent,
                'memory_warnings': self.memory_warnings,
                'memory_limit_mb': self.memory_limit_mb
            }
        except Exception as e:
            logging.warning(f"Error getting resource stats: {e}")
            return {}


class ProgressTracker:
    """Track and report batch processing progress."""
    
    def __init__(self, total_files: int, report_interval: int = 10):
        self.total_files = total_files
        self.report_interval = report_interval
        
        self.processed_files = 0
        self.failed_files = 0
        self.start_time = datetime.utcnow()
        self.last_report = self.start_time
        
        self._lock = threading.Lock()
    
    def update(self, processed: int = 0, failed: int = 0):
        """Update progress counters."""
        with self._lock:
            self.processed_files += processed
            self.failed_files += failed
            
            # Report progress if interval passed
            now = datetime.utcnow()
            if (now - self.last_report).seconds >= self.report_interval:
                self._report_progress()
                self.last_report = now
    
    def _report_progress(self):
        """Report current progress."""
        total_completed = self.processed_files + self.failed_files
        progress_percent = (total_completed / self.total_files) * 100 if self.total_files > 0 else 0
        
        elapsed = datetime.utcnow() - self.start_time
        rate = total_completed / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        remaining_files = self.total_files - total_completed
        eta_seconds = remaining_files / rate if rate > 0 else 0
        eta = timedelta(seconds=int(eta_seconds))
        
        logging.info(
            f"Progress: {total_completed}/{self.total_files} ({progress_percent:.1f}%) "
            f"- Processed: {self.processed_files}, Failed: {self.failed_files} "
            f"- Rate: {rate:.1f} files/sec, ETA: {eta}"
        )
    
    def get_final_report(self) -> Dict[str, Any]:
        """Get final progress report."""
        with self._lock:
            total_time = datetime.utcnow() - self.start_time
            total_completed = self.processed_files + self.failed_files
            
            return {
                'total_files': self.total_files,
                'processed_files': self.processed_files,
                'failed_files': self.failed_files,
                'total_completed': total_completed,
                'success_rate': self.processed_files / self.total_files if self.total_files > 0 else 0,
                'total_time': total_time.total_seconds(),
                'average_rate': total_completed / total_time.total_seconds() if total_time.total_seconds() > 0 else 0
            }


class BatchProcessor:
    """
    Main batch processor for handling large-scale SRT processing operations.
    
    Features:
    - Parallel processing with configurable worker pools
    - Resource monitoring and management
    - Progress tracking and reporting
    - Error handling and recovery
    - Performance metrics collection
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.metrics_collector = MetricsCollector()
        self.recovery_manager = RecoveryManager()
        
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # State
        self.is_processing = False
        self.current_batch_id = None
        
    def process_files_batch(
        self, 
        file_paths: List[str],
        processor_func: Callable,
        batch_id: str = None,
        **kwargs
    ) -> ProcessingResult:
        """
        Process a batch of files using parallel processing.
        
        Args:
            file_paths: List of file paths to process
            processor_func: Function to process individual files
            batch_id: Optional batch identifier
            **kwargs: Additional arguments for processor function
            
        Returns:
            ProcessingResult with processing statistics
        """
        if not file_paths:
            raise ValueError("No files provided for processing")
            
        batch_id = batch_id or f"batch_{int(time.time())}"
        self.current_batch_id = batch_id
        self.is_processing = True
        
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting batch processing: {batch_id} with {len(file_paths)} files")
            
            # Initialize monitoring and tracking
            resource_monitor = ResourceMonitor(
                memory_limit_mb=self.config.memory_limit_mb,
                check_interval=5.0
            )
            progress_tracker = ProgressTracker(
                total_files=len(file_paths),
                report_interval=self.config.progress_interval
            )
            
            # Start monitoring
            resource_monitor.start_monitoring()
            
            # Split files into chunks
            chunks = self._create_chunks(file_paths, self.config.chunk_size)
            
            # Process chunks in parallel
            results = self._process_chunks_parallel(
                chunks, 
                processor_func, 
                progress_tracker,
                **kwargs
            )
            
            # Stop monitoring
            resource_monitor.stop_monitoring()
            
            # Aggregate results
            end_time = datetime.utcnow()
            processing_result = self._aggregate_results(
                results, 
                batch_id, 
                len(file_paths),
                start_time,
                end_time
            )
            
            # Add resource statistics
            processing_result.metrics['resources'] = resource_monitor.get_current_stats()
            processing_result.metrics['progress'] = progress_tracker.get_final_report()
            
            # Collect metrics
            self.metrics_collector.record_batch_processing(processing_result)
            
            self.logger.info(
                f"Batch processing completed: {batch_id} - "
                f"{processing_result.processed_files}/{processing_result.total_files} files processed "
                f"({processing_result.success_rate:.1%} success rate)"
            )
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Error in batch processing {batch_id}: {e}")
            
            # Create error result
            end_time = datetime.utcnow()
            return ProcessingResult(
                batch_id=batch_id,
                total_files=len(file_paths),
                processed_files=0,
                failed_files=len(file_paths),
                processing_time=(end_time - start_time).total_seconds(),
                throughput=0.0,
                success_rate=0.0,
                errors=[{'batch_error': str(e)}],
                metrics={},
                start_time=start_time,
                end_time=end_time
            )
            
        finally:
            self.is_processing = False
            self.current_batch_id = None
    
    def _create_chunks(self, file_paths: List[str], chunk_size: int) -> List[List[str]]:
        """Split file paths into processing chunks."""
        chunks = []
        for i in range(0, len(file_paths), chunk_size):
            chunk = file_paths[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
    
    def _process_chunks_parallel(
        self,
        chunks: List[List[str]],
        processor_func: Callable,
        progress_tracker: ProgressTracker,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process chunks in parallel using ProcessPoolExecutor."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(
                    self._process_chunk,
                    chunk,
                    processor_func,
                    i,
                    **kwargs
                ): (chunk, i)
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk, timeout=self.config.timeout_seconds):
                chunk, chunk_id = future_to_chunk[future]
                
                try:
                    chunk_result = future.result()
                    results.append(chunk_result)
                    
                    # Update progress
                    progress_tracker.update(
                        processed=chunk_result['processed'],
                        failed=chunk_result['failed']
                    )
                    
                    self.logger.debug(
                        f"Chunk {chunk_id} completed: {chunk_result['processed']} processed, "
                        f"{chunk_result['failed']} failed"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Chunk {chunk_id} failed: {e}")
                    
                    # Create error result for chunk
                    error_result = {
                        'chunk_id': chunk_id,
                        'processed': 0,
                        'failed': len(chunk),
                        'processing_time': 0.0,
                        'errors': [{'chunk_error': str(e), 'files': chunk}]
                    }
                    results.append(error_result)
                    
                    # Update progress
                    progress_tracker.update(failed=len(chunk))
        
        return results
    
    @staticmethod
    def _process_chunk(
        file_chunk: List[str],
        processor_func: Callable,
        chunk_id: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single chunk of files."""
        start_time = time.time()
        
        result = {
            'chunk_id': chunk_id,
            'processed': 0,
            'failed': 0,
            'processing_time': 0.0,
            'errors': []
        }
        
        try:
            for file_path in file_chunk:
                try:
                    # Process individual file
                    processor_func(file_path, **kwargs)
                    result['processed'] += 1
                    
                except Exception as e:
                    result['failed'] += 1
                    result['errors'].append({
                        'file': file_path,
                        'error': str(e)
                    })
                    
                    # Log individual file errors
                    logging.warning(f"Failed to process {file_path}: {e}")
            
            result['processing_time'] = time.time() - start_time
            return result
            
        except Exception as e:
            # Chunk-level error
            result['failed'] = len(file_chunk)
            result['processing_time'] = time.time() - start_time
            result['errors'] = [{'chunk_error': str(e)}]
            return result
    
    def _aggregate_results(
        self,
        chunk_results: List[Dict[str, Any]],
        batch_id: str,
        total_files: int,
        start_time: datetime,
        end_time: datetime
    ) -> ProcessingResult:
        """Aggregate results from all chunks."""
        
        # Sum up metrics
        total_processed = sum(r['processed'] for r in chunk_results)
        total_failed = sum(r['failed'] for r in chunk_results)
        total_processing_time = (end_time - start_time).total_seconds()
        
        # Collect all errors
        all_errors = []
        for result in chunk_results:
            all_errors.extend(result.get('errors', []))
        
        # Calculate metrics
        throughput = total_processed / total_processing_time if total_processing_time > 0 else 0
        success_rate = total_processed / total_files if total_files > 0 else 0
        
        # Additional metrics
        metrics = {
            'chunk_count': len(chunk_results),
            'avg_chunk_time': sum(r['processing_time'] for r in chunk_results) / len(chunk_results),
            'max_chunk_time': max(r['processing_time'] for r in chunk_results),
            'min_chunk_time': min(r['processing_time'] for r in chunk_results),
            'total_errors': len(all_errors)
        }
        
        return ProcessingResult(
            batch_id=batch_id,
            total_files=total_files,
            processed_files=total_processed,
            failed_files=total_failed,
            processing_time=total_processing_time,
            throughput=throughput,
            success_rate=success_rate,
            errors=all_errors,
            metrics=metrics,
            start_time=start_time,
            end_time=end_time
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            'is_processing': self.is_processing,
            'current_batch_id': self.current_batch_id,
            'config': {
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'chunk_size': self.config.chunk_size,
                'memory_limit_mb': self.config.memory_limit_mb
            }
        }


# Utility functions for common batch processing scenarios

def process_srt_files_batch(
    input_dir: str,
    output_dir: str,
    processor_class,
    batch_size: int = 50,
    max_workers: int = None
) -> ProcessingResult:
    """
    Convenience function to batch process SRT files.
    
    Args:
        input_dir: Directory containing input SRT files
        output_dir: Directory for processed output files
        processor_class: SRT processor class to use
        batch_size: Number of files per batch
        max_workers: Number of parallel workers
        
    Returns:
        ProcessingResult with processing statistics
    """
    import glob
    
    # Find all SRT files
    pattern = os.path.join(input_dir, "**/*.srt")
    file_paths = glob.glob(pattern, recursive=True)
    
    if not file_paths:
        raise ValueError(f"No SRT files found in {input_dir}")
    
    # Configure processor
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers or mp.cpu_count()
    )
    
    batch_processor = BatchProcessor(config)
    
    def process_file(input_path: str, **kwargs):
        """Process individual SRT file."""
        # Calculate output path
        relative_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process file
        processor = processor_class()
        processor.process_srt_file(input_path, output_path)
    
    return batch_processor.process_files_batch(
        file_paths,
        process_file,
        batch_id=f"srt_batch_{int(time.time())}"
    )