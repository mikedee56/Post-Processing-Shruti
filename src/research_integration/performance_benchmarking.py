"""
Performance Benchmarking Framework

Provides comprehensive performance analysis comparing enhanced algorithms against baseline
implementations, with accuracy metrics calculation and processing time analysis.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import json
import statistics
from enum import Enum

from ..utils.logger_config import get_logger
from ..utils.srt_parser import SRTParser, SRTSegment
from ..post_processors.sanskrit_post_processor import SanskritPostProcessor

logger = get_logger(__name__)


class BenchmarkMetric(Enum):
    """Supported benchmark metrics"""
    WORD_ERROR_RATE = "word_error_rate"
    CHARACTER_ERROR_RATE = "character_error_rate"
    PROCESSING_TIME = "processing_time"
    CONFIDENCE_IMPROVEMENT = "confidence_improvement"
    ACCURACY_GAIN = "accuracy_gain"


@dataclass
class ProcessingResult:
    """Result of text processing operation"""
    original_text: str
    processed_text: str
    processing_time: float
    confidence_score: float
    changes_made: int
    algorithm_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkComparison:
    """Comparison between baseline and enhanced algorithm results"""
    baseline_result: ProcessingResult
    enhanced_result: ProcessingResult
    word_error_rate_reduction: float
    character_error_rate_reduction: float
    processing_time_ratio: float
    confidence_improvement: float
    accuracy_gain: float
    is_improvement: bool


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark analysis report"""
    test_name: str
    total_segments_tested: int
    baseline_algorithm: str
    enhanced_algorithm: str
    overall_accuracy_improvement: float
    avg_processing_time_ratio: float
    avg_confidence_improvement: float
    significant_improvements: int
    regressions_detected: int
    detailed_comparisons: List[BenchmarkComparison] = field(default_factory=list)
    performance_targets_met: bool = True
    summary_metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceBenchmarking:
    """
    Comprehensive performance benchmarking framework for research algorithm validation.
    
    Provides baseline vs enhanced algorithm comparison with accuracy metrics calculation
    and processing time analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance targets from architecture requirements
        self.performance_targets = {
            'max_processing_time_ratio': 2.0,  # <2x processing time increase
            'min_accuracy_improvement': 0.05,  # Minimum 5% accuracy improvement
            'min_confidence_improvement': 0.1,  # Minimum 10% confidence improvement
        }
        
        self.srt_parser = SRTParser()
        
    def create_baseline_processor(self) -> SanskritPostProcessor:
        """Create baseline processor with minimal enhancements"""
        baseline_config = {
            'enable_advanced_corrections': False,
            'enable_contextual_modeling': False,
            'enable_semantic_validation': False,
            'enable_sandhi_preprocessing': False
        }
        return SanskritPostProcessor(baseline_config)
    
    def create_enhanced_processor(self) -> SanskritPostProcessor:
        """Create enhanced processor with all research algorithms enabled"""
        enhanced_config = {
            'enable_advanced_corrections': True,
            'enable_contextual_modeling': True,
            'enable_semantic_validation': True,
            'enable_sandhi_preprocessing': True,
            'enable_hybrid_matching': True
        }
        return SanskritPostProcessor(enhanced_config)
    
    def calculate_word_error_rate(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER) between reference and hypothesis text.
        
        WER = (S + D + I) / N
        Where S=substitutions, D=deletions, I=insertions, N=total words in reference
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words:
            return 0.0 if not hyp_words else float('inf')
        
        # Simple Levenshtein distance calculation for words
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + cost  # substitution
                )
        
        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
        return min(wer, 1.0)  # Cap at 100%
    
    def calculate_character_error_rate(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER) between reference and hypothesis text.
        """
        ref_chars = reference.lower().replace(' ', '')
        hyp_chars = hypothesis.lower().replace(' ', '')
        
        if not ref_chars:
            return 0.0 if not hyp_chars else float('inf')
        
        # Character-level Levenshtein distance
        d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
        
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(hyp_chars) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                cost = 0 if ref_chars[i-1] == hyp_chars[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + cost  # substitution
                )
        
        cer = d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)
        return min(cer, 1.0)  # Cap at 100%
    
    def process_with_timing(self, processor: SanskritPostProcessor, segment: SRTSegment, 
                           algorithm_name: str) -> ProcessingResult:
        """Process a segment with timing and result collection"""
        start_time = time.time()
        
        try:
            # Create temporary SRT file for processing
            temp_content = f"1\n{segment.start_time} --> {segment.end_time}\n{segment.text}\n"
            
            # Count initial changes for comparison
            original_text = segment.text
            
            # Process the segment directly
            processed_segment = processor._process_srt_segment(
                segment, processor.metrics_collector.create_file_metrics("benchmark_test")
            )
            
            processing_time = time.time() - start_time
            processed_text = processed_segment.text
            
            # Calculate confidence score (simplified)
            confidence_score = 0.8 if original_text != processed_text else 0.6
            changes_made = 1 if original_text != processed_text else 0
            
            return ProcessingResult(
                original_text=original_text,
                processed_text=processed_text,
                processing_time=processing_time,
                confidence_score=confidence_score,
                changes_made=changes_made,
                algorithm_used=algorithm_name
            )
            
        except Exception as e:
            self.logger.error(f"Processing error with {algorithm_name}: {e}")
            
            # Return fallback result
            processing_time = time.time() - start_time
            return ProcessingResult(
                original_text=segment.text,
                processed_text=segment.text,
                processing_time=processing_time,
                confidence_score=0.0,
                changes_made=0,
                algorithm_used=f"{algorithm_name}_fallback",
                metadata={"error": str(e)}
            )
    
    def compare_algorithms(self, baseline_result: ProcessingResult, 
                          enhanced_result: ProcessingResult,
                          reference_text: Optional[str] = None) -> BenchmarkComparison:
        """
        Compare results between baseline and enhanced algorithms.
        
        Args:
            baseline_result: Result from baseline algorithm
            enhanced_result: Result from enhanced algorithm  
            reference_text: Optional golden reference for accuracy calculation
        """
        # Use original text as reference if no golden reference provided
        ref_text = reference_text or baseline_result.original_text
        
        # Calculate error rates
        baseline_wer = self.calculate_word_error_rate(ref_text, baseline_result.processed_text)
        enhanced_wer = self.calculate_word_error_rate(ref_text, enhanced_result.processed_text)
        
        baseline_cer = self.calculate_character_error_rate(ref_text, baseline_result.processed_text)
        enhanced_cer = self.calculate_character_error_rate(ref_text, enhanced_result.processed_text)
        
        # Calculate improvements
        wer_reduction = baseline_wer - enhanced_wer
        cer_reduction = baseline_cer - enhanced_cer
        
        processing_time_ratio = (enhanced_result.processing_time / 
                               max(baseline_result.processing_time, 0.0001))
        
        confidence_improvement = enhanced_result.confidence_score - baseline_result.confidence_score
        
        # Calculate overall accuracy gain
        accuracy_gain = (wer_reduction + cer_reduction) / 2.0
        
        # Determine if this represents an improvement
        is_improvement = (
            accuracy_gain > 0 and 
            processing_time_ratio <= self.performance_targets['max_processing_time_ratio']
        )
        
        return BenchmarkComparison(
            baseline_result=baseline_result,
            enhanced_result=enhanced_result,
            word_error_rate_reduction=wer_reduction,
            character_error_rate_reduction=cer_reduction,
            processing_time_ratio=processing_time_ratio,
            confidence_improvement=confidence_improvement,
            accuracy_gain=accuracy_gain,
            is_improvement=is_improvement
        )
    
    def benchmark_srt_processing(self, srt_file_path: Path, 
                               max_segments: Optional[int] = None,
                               golden_dataset: Optional[Dict[str, str]] = None) -> BenchmarkReport:
        """
        Comprehensive benchmark of SRT processing comparing baseline vs enhanced algorithms.
        
        Args:
            srt_file_path: Path to SRT file for testing
            max_segments: Maximum segments to test (for performance)
            golden_dataset: Optional golden reference texts for accuracy calculation
        """
        self.logger.info(f"Starting benchmark of {srt_file_path}")
        
        # Parse SRT file
        segments = self.srt_parser.parse_file(srt_file_path)
        if max_segments:
            segments = segments[:max_segments]
        
        self.logger.info(f"Benchmarking {len(segments)} segments")
        
        # Create processors
        baseline_processor = self.create_baseline_processor()
        enhanced_processor = self.create_enhanced_processor()
        
        comparisons = []
        processing_time_ratios = []
        accuracy_gains = []
        confidence_improvements = []
        improvements = 0
        regressions = 0
        
        for i, segment in enumerate(segments):
            self.logger.debug(f"Processing segment {i+1}/{len(segments)}")
            
            # Process with both algorithms
            baseline_result = self.process_with_timing(baseline_processor, segment, "baseline")
            enhanced_result = self.process_with_timing(enhanced_processor, segment, "enhanced")
            
            # Get golden reference if available
            golden_ref = golden_dataset.get(str(i)) if golden_dataset else None
            
            # Compare results
            comparison = self.compare_algorithms(baseline_result, enhanced_result, golden_ref)
            comparisons.append(comparison)
            
            # Collect metrics
            processing_time_ratios.append(comparison.processing_time_ratio)
            accuracy_gains.append(comparison.accuracy_gain)
            confidence_improvements.append(comparison.confidence_improvement)
            
            if comparison.is_improvement:
                improvements += 1
            elif comparison.accuracy_gain < -0.01:  # Significant regression
                regressions += 1
        
        # Calculate overall metrics
        overall_accuracy_improvement = statistics.mean(accuracy_gains) if accuracy_gains else 0.0
        avg_processing_time_ratio = statistics.mean(processing_time_ratios) if processing_time_ratios else 1.0
        avg_confidence_improvement = statistics.mean(confidence_improvements) if confidence_improvements else 0.0
        
        # Check performance targets
        performance_targets_met = (
            avg_processing_time_ratio <= self.performance_targets['max_processing_time_ratio'] and
            overall_accuracy_improvement >= self.performance_targets['min_accuracy_improvement']
        )
        
        # Create comprehensive report
        report = BenchmarkReport(
            test_name=f"SRT_Benchmark_{srt_file_path.stem}",
            total_segments_tested=len(segments),
            baseline_algorithm="sanskrit_post_processor_baseline",
            enhanced_algorithm="sanskrit_post_processor_enhanced",
            overall_accuracy_improvement=overall_accuracy_improvement,
            avg_processing_time_ratio=avg_processing_time_ratio,
            avg_confidence_improvement=avg_confidence_improvement,
            significant_improvements=improvements,
            regressions_detected=regressions,
            detailed_comparisons=comparisons,
            performance_targets_met=performance_targets_met,
            summary_metrics={
                'wer_reduction_avg': statistics.mean([c.word_error_rate_reduction for c in comparisons]),
                'cer_reduction_avg': statistics.mean([c.character_error_rate_reduction for c in comparisons]),
                'processing_time_std': statistics.stdev(processing_time_ratios) if len(processing_time_ratios) > 1 else 0.0,
                'accuracy_gain_std': statistics.stdev(accuracy_gains) if len(accuracy_gains) > 1 else 0.0
            }
        )
        
        self.logger.info(f"Benchmark completed: {improvements} improvements, {regressions} regressions")
        return report
    
    def export_benchmark_report(self, report: BenchmarkReport, output_path: Path) -> None:
        """Export benchmark report to JSON file"""
        try:
            report_data = {
                'test_name': report.test_name,
                'timestamp': time.time(),
                'total_segments_tested': report.total_segments_tested,
                'baseline_algorithm': report.baseline_algorithm,
                'enhanced_algorithm': report.enhanced_algorithm,
                'overall_accuracy_improvement': report.overall_accuracy_improvement,
                'avg_processing_time_ratio': report.avg_processing_time_ratio,
                'avg_confidence_improvement': report.avg_confidence_improvement,
                'significant_improvements': report.significant_improvements,
                'regressions_detected': report.regressions_detected,
                'performance_targets_met': report.performance_targets_met,
                'summary_metrics': report.summary_metrics,
                'performance_targets': self.performance_targets
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Benchmark report exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export benchmark report: {e}")
            raise
    
    def validate_performance_targets(self, report: BenchmarkReport) -> Dict[str, bool]:
        """
        Validate benchmark results against performance targets.
        
        Returns dict of target validation results.
        """
        validations = {
            'processing_time_target': report.avg_processing_time_ratio <= self.performance_targets['max_processing_time_ratio'],
            'accuracy_improvement_target': report.overall_accuracy_improvement >= self.performance_targets['min_accuracy_improvement'],
            'confidence_improvement_target': report.avg_confidence_improvement >= self.performance_targets['min_confidence_improvement'],
            'regression_tolerance': report.regressions_detected <= max(1, report.total_segments_tested * 0.05)  # Max 5% regressions
        }
        
        validations['overall_pass'] = all(validations.values())
        
        return validations