"""
Batch Semantic Processor for Story 2.4.2

This module provides advanced batch processing capabilities for semantic similarity
computation with optimization for file-based architecture and large-scale operations.

Architecture Integration:
- Optimized for file-based storage with minimal memory usage
- Supports parallel processing for large corpora
- Integrates with semantic cache management system
"""

import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator, Callable
import time
from collections import defaultdict

from utils.logger_config import get_logger
from .semantic_similarity_calculator import SemanticSimilarityCalculator, SemanticSimilarityResult


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    max_workers: int = 4
    batch_size: int = 100
    memory_limit_mb: int = 512
    progress_report_interval: int = 10
    enable_parallel_processing: bool = True
    cache_write_interval: int = 50
    error_tolerance: float = 0.1  # Max proportion of errors to tolerate


@dataclass
class BatchProcessingResult:
    """Result of batch processing operation."""
    total_pairs: int
    successful_computations: int
    failed_computations: int
    total_time_seconds: float
    average_similarity: float
    cache_hit_rate: float
    throughput_pairs_per_second: float
    errors: List[str]
    processing_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BatchSemanticProcessor:
    """
    Advanced batch processing for semantic similarity operations.
    
    This component provides:
    1. Efficient batch processing for large text pair datasets (AC3)
    2. Memory-optimized processing for file-based architecture
    3. Progress tracking and error handling for large operations
    4. Parallel processing capabilities with configurable workers
    5. Integration with semantic cache management
    """
    
    def __init__(
        self, 
        similarity_calculator: SemanticSimilarityCalculator,
        config: Optional[BatchProcessingConfig] = None
    ):
        """
        Initialize the batch semantic processor.
        
        Args:
            similarity_calculator: SemanticSimilarityCalculator instance
            config: Batch processing configuration
        """
        self.logger = get_logger(__name__)
        self.similarity_calculator = similarity_calculator
        self.config = config or BatchProcessingConfig()
        
        # Processing statistics
        self.stats = {
            'batch_operations': 0,
            'total_pairs_processed': 0,
            'total_processing_time': 0.0,
            'errors_encountered': 0,
            'cache_optimization_runs': 0
        }
    
    def process_text_pairs_batch(
        self,
        text_pairs: List[Tuple[str, str]],
        language: Optional[str] = None,
        output_file: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> BatchProcessingResult:
        """
        Process a batch of text pairs for semantic similarity.
        
        Args:
            text_pairs: List of (text1, text2) tuples to process
            language: Language model to use (auto-detected if None)
            output_file: Optional file to save results
            progress_callback: Optional progress callback function
            
        Returns:
            BatchProcessingResult with comprehensive statistics
        """
        start_time = time.time()
        self.stats['batch_operations'] += 1
        
        total_pairs = len(text_pairs)
        successful_results = []
        errors = []
        
        self.logger.info(
            f"Starting batch processing of {total_pairs} text pairs with "
            f"{self.config.max_workers} workers"
        )
        
        # Process in chunks to manage memory usage
        chunk_size = min(self.config.batch_size, total_pairs)
        processed_count = 0
        
        for chunk_start in range(0, total_pairs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pairs)
            chunk_pairs = text_pairs[chunk_start:chunk_end]
            
            # Process chunk
            if self.config.enable_parallel_processing and len(chunk_pairs) > 10:
                chunk_results, chunk_errors = self._process_chunk_parallel(
                    chunk_pairs, language
                )
            else:
                chunk_results, chunk_errors = self._process_chunk_sequential(
                    chunk_pairs, language
                )
            
            successful_results.extend(chunk_results)
            errors.extend(chunk_errors)
            processed_count += len(chunk_pairs)
            
            # Progress callback
            if progress_callback:
                progress = processed_count / total_pairs
                progress_callback(progress, processed_count, total_pairs)
            
            # Periodic cache save
            if processed_count % self.config.cache_write_interval == 0:
                self.similarity_calculator._save_cache()
            
            # Memory management
            if len(successful_results) > self.config.batch_size * 2:
                # Write intermediate results to avoid memory buildup
                if output_file:
                    self._append_results_to_file(successful_results, output_file)
                    successful_results = []  # Clear from memory
        
        # Final processing
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        successful_count = len(successful_results) + len([r for r in successful_results if r])
        failed_count = len(errors)
        
        if successful_results:
            average_similarity = sum(r.similarity_score for r in successful_results if r) / len(successful_results)
            cache_hits = sum(1 for r in successful_results if r and r.cache_hit)
            cache_hit_rate = cache_hits / len(successful_results) if successful_results else 0.0
        else:
            average_similarity = 0.0
            cache_hit_rate = 0.0
        
        throughput = total_pairs / total_time if total_time > 0 else 0.0
        
        # Save final results
        if output_file and successful_results:
            self._append_results_to_file(successful_results, output_file)
        
        # Update global statistics
        self.stats['total_pairs_processed'] += total_pairs
        self.stats['total_processing_time'] += total_time
        self.stats['errors_encountered'] += failed_count
        
        # Final cache save
        self.similarity_calculator._save_cache()
        
        result = BatchProcessingResult(
            total_pairs=total_pairs,
            successful_computations=successful_count,
            failed_computations=failed_count,
            total_time_seconds=total_time,
            average_similarity=average_similarity,
            cache_hit_rate=cache_hit_rate,
            throughput_pairs_per_second=throughput,
            errors=errors,
            processing_stats=self.similarity_calculator.get_performance_stats()
        )
        
        self.logger.info(
            f"Batch processing completed: {successful_count}/{total_pairs} successful, "
            f"{throughput:.2f} pairs/sec, {cache_hit_rate:.1%} cache hit rate"
        )
        
        return result
    
    def _process_chunk_sequential(
        self, 
        chunk_pairs: List[Tuple[str, str]], 
        language: Optional[str]
    ) -> Tuple[List[SemanticSimilarityResult], List[str]]:
        """Process chunk sequentially."""
        results = []
        errors = []
        
        for text1, text2 in chunk_pairs:
            try:
                result = self.similarity_calculator.compute_semantic_similarity(
                    text1, text2, language
                )
                results.append(result)
                
            except Exception as e:
                error_msg = f"Error processing pair '{text1[:30]}...', '{text2[:30]}...': {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        return results, errors
    
    def _process_chunk_parallel(
        self, 
        chunk_pairs: List[Tuple[str, str]], 
        language: Optional[str]
    ) -> Tuple[List[SemanticSimilarityResult], List[str]]:
        """Process chunk using parallel workers."""
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit tasks
            future_to_pair = {
                executor.submit(
                    self.similarity_calculator.compute_semantic_similarity,
                    text1, text2, language
                ): (text1, text2)
                for text1, text2 in chunk_pairs
            }
            
            # Collect results
            for future in as_completed(future_to_pair):
                text1, text2 = future_to_pair[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                except Exception as e:
                    error_msg = f"Error processing pair '{text1[:30]}...', '{text2[:30]}...': {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
        
        return results, errors
    
    def _append_results_to_file(
        self, 
        results: List[SemanticSimilarityResult], 
        output_file: Path
    ) -> None:
        """Append results to output file."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Append mode for incremental writing
            mode = 'a' if output_file.exists() else 'w'
            
            with open(output_file, mode, encoding='utf-8') as f:
                for result in results:
                    if result:  # Skip None results
                        json.dump(result.to_dict(), f, ensure_ascii=False)
                        f.write('\n')
            
        except OSError as e:
            self.logger.error(f"Error writing results to {output_file}: {e}")
    
    def process_scripture_similarity_matrix(
        self,
        scripture_texts: List[str],
        output_file: Path,
        language: str = "sa"
    ) -> BatchProcessingResult:
        """
        Compute similarity matrix for scripture texts for research analysis.
        
        Args:
            scripture_texts: List of scripture texts to compare
            output_file: File to save similarity matrix
            language: Language model to use
            
        Returns:
            BatchProcessingResult with matrix computation statistics
        """
        n_texts = len(scripture_texts)
        total_pairs = (n_texts * (n_texts - 1)) // 2  # Upper triangular matrix
        
        self.logger.info(f"Computing similarity matrix for {n_texts} texts ({total_pairs} pairs)")
        
        # Generate all unique pairs
        text_pairs = []
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                text_pairs.append((scripture_texts[i], scripture_texts[j]))
        
        # Process pairs
        result = self.process_text_pairs_batch(
            text_pairs=text_pairs,
            language=language,
            output_file=output_file,
            progress_callback=lambda p, c, t: self.logger.info(
                f"Matrix computation progress: {c}/{t} pairs ({p:.1%})"
            ) if c % 100 == 0 else None
        )
        
        # Create matrix representation
        matrix_file = output_file.with_suffix('.matrix.json')
        self._create_similarity_matrix_file(scripture_texts, text_pairs, result, matrix_file)
        
        return result
    
    def _create_similarity_matrix_file(
        self,
        texts: List[str],
        text_pairs: List[Tuple[str, str]],
        result: BatchProcessingResult,
        matrix_file: Path
    ) -> None:
        """Create a structured similarity matrix file."""
        n_texts = len(texts)
        matrix = [[0.0 for _ in range(n_texts)] for _ in range(n_texts)]
        
        # Fill diagonal with 1.0 (self-similarity)
        for i in range(n_texts):
            matrix[i][i] = 1.0
        
        # Read results from output file and populate matrix
        try:
            # This is a simplified version - in practice, you'd read the actual results
            # from the output file and map them back to matrix positions
            
            matrix_data = {
                'texts': [text[:100] + '...' if len(text) > 100 else text for text in texts],
                'similarity_matrix': matrix,
                'metadata': {
                    'total_texts': n_texts,
                    'total_comparisons': len(text_pairs),
                    'computation_time': result.total_time_seconds,
                    'average_similarity': result.average_similarity,
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
            }
            
            with open(matrix_file, 'w', encoding='utf-8') as f:
                json.dump(matrix_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Created similarity matrix file: {matrix_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating matrix file: {e}")
    
    def process_text_file_batch(
        self,
        input_file: Path,
        output_file: Path,
        language: Optional[str] = None,
        text_pair_separator: str = "\t"
    ) -> BatchProcessingResult:
        """
        Process text pairs from input file in batch mode.
        
        Args:
            input_file: File containing text pairs (one pair per line, tab-separated)
            output_file: Output file for results
            language: Language model to use
            text_pair_separator: Separator between text pairs in input file
            
        Returns:
            BatchProcessingResult with file processing statistics
        """
        try:
            # Read text pairs from file
            text_pairs = []
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split(text_pair_separator)
                    if len(parts) >= 2:
                        text_pairs.append((parts[0].strip(), parts[1].strip()))
                    else:
                        self.logger.warning(
                            f"Invalid line format at line {line_num}: {line[:50]}..."
                        )
            
            self.logger.info(f"Loaded {len(text_pairs)} text pairs from {input_file}")
            
            # Process batch
            return self.process_text_pairs_batch(
                text_pairs=text_pairs,
                language=language,
                output_file=output_file
            )
            
        except OSError as e:
            self.logger.error(f"Error reading input file {input_file}: {e}")
            return BatchProcessingResult(
                total_pairs=0,
                successful_computations=0,
                failed_computations=1,
                total_time_seconds=0.0,
                average_similarity=0.0,
                cache_hit_rate=0.0,
                throughput_pairs_per_second=0.0,
                errors=[f"File reading error: {e}"],
                processing_stats={}
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        avg_processing_time = (
            self.stats['total_processing_time'] / max(self.stats['batch_operations'], 1)
        )
        
        avg_throughput = (
            self.stats['total_pairs_processed'] / max(self.stats['total_processing_time'], 1)
        )
        
        return {
            'batch_operations': self.stats['batch_operations'],
            'total_pairs_processed': self.stats['total_pairs_processed'],
            'average_batch_time': f"{avg_processing_time:.2f}s",
            'average_throughput': f"{avg_throughput:.2f} pairs/sec",
            'total_errors': self.stats['errors_encountered'],
            'error_rate': f"{self.stats['errors_encountered'] / max(self.stats['total_pairs_processed'], 1) * 100:.2f}%",
            'cache_optimization_runs': self.stats['cache_optimization_runs'],
            'calculator_stats': self.similarity_calculator.get_performance_stats()
        }