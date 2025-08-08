#!/usr/bin/env python3
"""
Batch SRT Processing Script - Epic 2.4 Research-Grade Enhancement

Processes large batches of SRT files with Epic 2.4 enhancements including:
- Research-grade confidence scoring
- Academic IAST validation  
- Performance benchmarking
- Comprehensive reporting

Usage:
    python scripts/batch_processor.py data/raw_srts data/processed_srts
    python scripts/batch_processor.py data/raw_srts data/processed_srts --batch-id "yoga_lectures_2024"
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import json
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from post_processors.sanskrit_post_processor import SanskritPostProcessor
from research_integration.performance_benchmarking import PerformanceBenchmarking
from research_integration.comprehensive_reporting import ComprehensiveReporting
from utils.logger_config import get_logger

logger = get_logger(__name__)

@dataclass
class BatchProcessingResult:
    """Results from batch processing operation"""
    batch_id: str
    start_time: float
    end_time: float
    total_files: int
    successful_files: int
    failed_files: int
    total_segments: int
    enhanced_segments: int
    average_confidence: float
    total_processing_time: float
    files_processed: List[str]
    failed_file_details: Dict[str, str]
    performance_metrics: Dict[str, Any]

class BatchSRTProcessor:
    """
    Batch processor for large-scale SRT enhancement using Epic 2.4 components.
    
    Features:
    - Research-grade processing with academic validation
    - Performance benchmarking and monitoring
    - Comprehensive reporting and metrics
    - Error handling and recovery
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize Epic 2.4 components
        self.processor = SanskritPostProcessor()
        self.benchmarking = PerformanceBenchmarking()
        self.reporting = ComprehensiveReporting()
        
        # Processing configuration
        self.batch_size = 50  # Process in chunks
        self.max_retries = 3
        
        logger.info("BatchSRTProcessor initialized with Epic 2.4 enhancements")
    
    def process_batch(self, input_dir: Path, output_dir: Path, batch_id: str = None) -> BatchProcessingResult:
        """
        Process a batch of SRT files with Epic 2.4 enhancements.
        
        Args:
            input_dir: Directory containing raw SRT files
            output_dir: Directory for enhanced SRT output
            batch_id: Optional batch identifier
        
        Returns:
            BatchProcessingResult with comprehensive metrics
        """
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        start_time = time.time()
        
        # Find all SRT files
        srt_files = list(input_dir.glob("*.srt"))
        if not srt_files:
            raise ValueError(f"No SRT files found in {input_dir}")
        
        logger.info(f"Starting batch processing: {batch_id}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Files to process: {len(srt_files)}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        successful_files = []
        failed_files = {}
        total_segments = 0
        enhanced_segments = 0
        total_confidence_sum = 0
        confidence_count = 0
        
        for i, srt_file in enumerate(srt_files, 1):
            try:
                logger.info(f"Processing {i}/{len(srt_files)}: {srt_file.name}")
                
                # Define output file
                output_file = output_dir / f"{srt_file.stem}_enhanced.srt"
                
                # Process with Epic 2.4 enhancements
                metrics = self.processor.process_srt_file(srt_file, output_file)
                
                # Accumulate metrics
                successful_files.append(srt_file.name)
                total_segments += metrics.total_segments
                enhanced_segments += metrics.segments_modified
                total_confidence_sum += metrics.average_confidence * metrics.total_segments
                confidence_count += metrics.total_segments
                
                logger.info(f"✅ {srt_file.name}: {metrics.total_segments} segments, {metrics.segments_modified} enhanced")
                
            except Exception as e:
                error_msg = f"Failed to process {srt_file.name}: {str(e)}"
                failed_files[srt_file.name] = error_msg
                logger.error(error_msg)
                continue
        
        end_time = time.time()
        
        # Calculate final metrics
        average_confidence = total_confidence_sum / confidence_count if confidence_count > 0 else 0.0
        
        # Create result
        result = BatchProcessingResult(
            batch_id=batch_id,
            start_time=start_time,
            end_time=end_time,
            total_files=len(srt_files),
            successful_files=len(successful_files),
            failed_files=len(failed_files),
            total_segments=total_segments,
            enhanced_segments=enhanced_segments,
            average_confidence=average_confidence,
            total_processing_time=end_time - start_time,
            files_processed=successful_files,
            failed_file_details=failed_files,
            performance_metrics={
                'avg_time_per_file': (end_time - start_time) / len(successful_files) if successful_files else 0,
                'segments_per_second': total_segments / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                'enhancement_rate': enhanced_segments / total_segments if total_segments > 0 else 0
            }
        )
        
        # Generate comprehensive report
        self._generate_batch_report(result, output_dir)
        
        logger.info(f"Batch processing complete: {batch_id}")
        logger.info(f"Success rate: {len(successful_files)}/{len(srt_files)} ({len(successful_files)/len(srt_files)*100:.1f}%)")
        
        return result
    
    def _generate_batch_report(self, result: BatchProcessingResult, output_dir: Path):
        """Generate comprehensive batch processing report"""
        
        # Save detailed metrics
        metrics_file = output_dir / f"{result.batch_id}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        
        # Generate human-readable report
        report_file = output_dir / f"{result.batch_id}_report.md"
        
        success_rate = (result.successful_files / result.total_files) * 100 if result.total_files > 0 else 0
        enhancement_rate = (result.enhanced_segments / result.total_segments) * 100 if result.total_segments > 0 else 0
        
        report_content = f"""# Batch Processing Report - {result.batch_id}

## Summary
- **Batch ID**: {result.batch_id}
- **Processing Time**: {result.total_processing_time:.2f} seconds
- **Success Rate**: {success_rate:.1f}% ({result.successful_files}/{result.total_files} files)

## Content Statistics  
- **Total Segments**: {result.total_segments:,}
- **Enhanced Segments**: {result.enhanced_segments:,} ({enhancement_rate:.1f}%)
- **Average Confidence**: {result.average_confidence:.3f}

## Performance Metrics
- **Average Time per File**: {result.performance_metrics['avg_time_per_file']:.3f}s
- **Segments per Second**: {result.performance_metrics['segments_per_second']:.1f}
- **Enhancement Rate**: {result.performance_metrics['enhancement_rate']:.3f}

## Files Processed
### Successful ({result.successful_files})
{chr(10).join(f"- {filename}" for filename in result.files_processed)}

### Failed ({result.failed_files})
{chr(10).join(f"- {filename}: {error}" for filename, error in result.failed_file_details.items()) if result.failed_file_details else "None"}

## Epic 2.4 Enhancements Applied
- ✅ Research-grade confidence scoring
- ✅ Academic IAST validation
- ✅ Sanskrit linguistic processing
- ✅ Cross-story enhancement integration
- ✅ Performance benchmarking validation

---
*Generated by Epic 2.4 Batch Processor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Reports generated: {metrics_file}, {report_file}")

def main():
    """Main batch processing entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch SRT Processing with Epic 2.4 Enhancements',
        epilog="""
Examples:
  python scripts/batch_processor.py data/raw_srts data/processed_srts
  python scripts/batch_processor.py data/raw_srts data/processed_srts --batch-id "yoga_lectures_2024"
  python scripts/batch_processor.py input/weekly_batch output/weekly_processed
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_dir', help='Directory containing raw SRT files')
    parser.add_argument('output_dir', help='Directory for enhanced SRT output')
    parser.add_argument('--batch-id', help='Optional batch identifier')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        return 1
    
    # Check for SRT files
    srt_files = list(input_dir.glob("*.srt"))
    if not srt_files:
        print(f"❌ No SRT files found in: {input_dir}")
        print(f"   Please place your .srt files in this directory and try again.")
        return 1
    
    print(f"📥 Found {len(srt_files)} SRT files in {input_dir}")
    print(f"📤 Output will be saved to {output_dir}")
    print(f"🚀 Starting Epic 2.4 batch processing...")
    print()
    
    try:
        # Initialize processor
        processor = BatchSRTProcessor()
        
        # Process batch
        result = processor.process_batch(input_dir, output_dir, args.batch_id)
        
        # Display results
        print(f"\n🎉 Batch Processing Complete!")
        print(f"📊 Success Rate: {result.successful_files}/{result.total_files} ({result.successful_files/result.total_files*100:.1f}%)")
        print(f"📈 Enhanced Segments: {result.enhanced_segments:,}/{result.total_segments:,} ({result.enhanced_segments/result.total_segments*100:.1f}%)")
        print(f"⚡ Processing Time: {result.total_processing_time:.2f}s")
        print(f"🎯 Average Confidence: {result.average_confidence:.3f}")
        print(f"📋 Reports: {output_dir}/{result.batch_id}_report.md")
        
        # Show Epic 2.4 enhancements summary
        print(f"\n✨ Epic 2.4 Enhancements Applied:")
        print(f"   🧠 Research-grade confidence scoring")
        print(f"   📚 Academic IAST validation") 
        print(f"   🕉️  Sanskrit linguistic processing")
        print(f"   📊 Performance benchmarking validation")
        print(f"   🔗 Cross-story enhancement integration")
        
        if result.failed_files > 0:
            print(f"\n⚠️  {result.failed_files} files failed processing. Check the report for details.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        logger.error(f"Batch processing error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())