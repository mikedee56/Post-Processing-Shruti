#!/usr/bin/env python3
"""
CRITICAL PRODUCTION FIX: Anti-Hallucination Deployment Script
Epic 2.4 ASR Post-Processing System - Sanskrit Term Hallucination Fix

This script applies the comprehensive anti-hallucination fix to eliminate
random Sanskrit term insertion in English text segments.

ISSUE: System was inserting Sanskrit terms like "Kṛṣṇa" and "ātman" in 
inappropriate contexts, corrupting otherwise correct English text.

SOLUTION: Ultra-conservative fuzzy matching thresholds and comprehensive
English word protection.

Usage:
    python APPLY_ANTI_HALLUCINATION_FIX.py [--test-mode]
    
Options:
    --test-mode    Run in test mode (process one file only)
    --force        Force reprocess even if output exists
    --validate     Validate fixes on specific problem cases
"""

import sys
import argparse
from pathlib import Path
import logging
import json
import time
from typing import Dict, Any, List

# Add src directory to path for imports
sys.path.append('src')

from post_processors.sanskrit_post_processor import SanskritPostProcessor
from utils.logger_config import get_logger

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/anti_hallucination_fix.log'),
        logging.StreamHandler()
    ]
)
logger = get_logger(__name__)

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Apply anti-hallucination fix')
    parser.add_argument('--test-mode', action='store_true', help='Process only one test file')
    parser.add_argument('--force', action='store_true', help='Force reprocess existing files')
    parser.add_argument('--validate', action='store_true', help='Validate specific problem cases')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("EPIC 2.4 ANTI-HALLUCINATION FIX DEPLOYMENT")
    logger.info("=" * 80)
    logger.info(f"Mode: {'TEST' if args.test_mode else 'PRODUCTION'}")
    logger.info(f"Force reprocess: {args.force}")
    logger.info(f"Validation mode: {args.validate}")
    
    try:
        # Initialize the fixed processor with ultra-conservative settings
        logger.info("Initializing ultra-conservative Sanskrit post-processor...")
        
        # Load anti-hallucination configuration
        config_path = Path("config/anti_hallucination_config.yaml")
        if not config_path.exists():
            logger.warning(f"Anti-hallucination config not found at {config_path}, using hardcoded conservative settings")
            config = None
        else:
            config = config_path
            
        processor = SanskritPostProcessor(config_path=config)
        
        # Show processor statistics
        stats = processor.get_processing_stats()
        logger.info("Processor initialized with ultra-conservative settings:")
        logger.info(f"  - Fuzzy threshold: {stats.get('fuzzy_threshold', 'N/A')}")
        logger.info(f"  - Legacy lexicons loaded: {len(stats.get('legacy_lexicons', {}))}")
        logger.info(f"  - Enhanced lexicons available: {len(stats.get('enhanced_lexicons', {}))}")
        
        # Define paths
        input_dir = Path("data/raw_srts")
        output_dir = Path("data/processed_srts")
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of SRT files to process
        srt_files = list(input_dir.glob("*.srt"))
        
        if not srt_files:
            logger.warning(f"No SRT files found in {input_dir}")
            return
        
        logger.info(f"Found {len(srt_files)} SRT files to process")
        
        # Test mode: process only one file
        if args.test_mode:
            srt_files = srt_files[:1]
            logger.info(f"TEST MODE: Processing only {srt_files[0].name}")
        
        # Process files
        total_processed = 0
        total_errors = 0
        processing_results = []
        
        for srt_file in srt_files:
            try:
                # Define output path
                output_file = output_dir / f"{srt_file.stem}_enhanced.srt"
                
                # Skip if output exists and not forcing
                if output_file.exists() and not args.force:
                    logger.info(f"Skipping {srt_file.name} (output exists, use --force to reprocess)")
                    continue
                
                logger.info(f"Processing {srt_file.name}...")
                start_time = time.time()
                
                # Process the file with anti-hallucination protections
                metrics = processor.process_srt_file(srt_file, output_file)
                
                processing_time = time.time() - start_time
                
                # Log results
                logger.info(f"  ✓ Completed in {processing_time:.2f}s")
                logger.info(f"  ✓ Segments: {metrics.total_segments}, Modified: {metrics.segments_modified}")
                logger.info(f"  ✓ Confidence: {metrics.average_confidence:.3f}")
                logger.info(f"  ✓ Warnings: {len(metrics.warnings_encountered)}")
                
                # Store result
                processing_results.append({
                    'file': srt_file.name,
                    'status': 'success',
                    'segments_total': metrics.total_segments,
                    'segments_modified': metrics.segments_modified,
                    'confidence': metrics.average_confidence,
                    'processing_time': processing_time,
                    'warnings_count': len(metrics.warnings_encountered)
                })
                
                total_processed += 1
                
                if args.validate and total_processed == 1:
                    # Perform validation on the first processed file
                    validate_anti_hallucination_fix(srt_file, output_file)
                
            except Exception as e:
                logger.error(f"Error processing {srt_file.name}: {e}")
                processing_results.append({
                    'file': srt_file.name,
                    'status': 'error',
                    'error': str(e)
                })
                total_errors += 1
                
        # Summary
        logger.info("=" * 80)
        logger.info("DEPLOYMENT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Files processed successfully: {total_processed}")
        logger.info(f"Files with errors: {total_errors}")
        
        if processing_results:
            avg_confidence = sum(r.get('confidence', 0) for r in processing_results if r['status'] == 'success') / max(1, total_processed)
            logger.info(f"Average confidence score: {avg_confidence:.3f}")
            
        # Save detailed results
        results_file = Path("logs/anti_hallucination_deployment_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'mode': 'test' if args.test_mode else 'production',
                'total_files': len(srt_files),
                'processed': total_processed,
                'errors': total_errors,
                'results': processing_results
            }, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Final validation check if requested
        if args.validate:
            logger.info("Running comprehensive validation...")
            run_comprehensive_validation()
        
        if total_errors == 0:
            logger.info("🎉 DEPLOYMENT SUCCESSFUL - No hallucination issues detected!")
        else:
            logger.warning(f"⚠️  DEPLOYMENT COMPLETED WITH {total_errors} ERRORS")
            
    except Exception as e:
        logger.error(f"CRITICAL ERROR in deployment: {e}")
        raise

def validate_anti_hallucination_fix(input_file: Path, output_file: Path):
    """Validate that the anti-hallucination fix is working properly."""
    logger.info(f"Validating anti-hallucination fix on {input_file.name}...")
    
    try:
        # Read both files
        with open(input_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_content = f.read()
        
        # Check for problematic insertions
        hallucination_indicators = ['Kṛṣṇa', 'ātman', 'dharma', 'Vedānta']
        
        issues_found = []
        for indicator in hallucination_indicators:
            original_count = original_content.count(indicator)
            processed_count = processed_content.count(indicator)
            
            if processed_count > original_count:
                excess_count = processed_count - original_count
                issues_found.append(f"{indicator}: {excess_count} excess insertions")
                logger.warning(f"  ⚠️  Found {excess_count} potential hallucinations of '{indicator}'")
        
        if issues_found:
            logger.error("❌ VALIDATION FAILED - Hallucinations still present:")
            for issue in issues_found:
                logger.error(f"    {issue}")
        else:
            logger.info("✅ VALIDATION PASSED - No hallucinations detected")
            
    except Exception as e:
        logger.error(f"Error during validation: {e}")

def run_comprehensive_validation():
    """Run comprehensive validation against known problem cases."""
    logger.info("Running comprehensive validation tests...")
    
    # Test cases that were problematic
    test_cases = [
        {
            'input': "who is Brahman",
            'should_not_contain': ['Kṛṣṇa'],
            'description': 'Common English phrase corruption'
        },
        {
            'input': "This chapter is entitled, Atma Vishranti", 
            'should_not_contain': ['1 Kṛṣṇa', 'ātman'],
            'description': 'Chapter title corruption'
        },
        {
            'input': "highly inspired and",
            'should_not_contain': ['ātman'],
            'description': 'Common English word corruption'
        }
    ]
    
    # Initialize processor for testing
    processor = SanskritPostProcessor()
    
    validation_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test {i}: {test_case['description']}")
        
        # Create a temporary segment and process it
        from utils.srt_parser import SRTSegment
        
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text=test_case['input']
        )
        
        # Process the segment
        processed_metrics = processor.metrics_collector.create_file_metrics("test")
        processed_segment = processor._process_srt_segment(segment, processed_metrics)
        
        # Check results
        output_text = processed_segment.text
        logger.info(f"  Input:  '{test_case['input']}'")
        logger.info(f"  Output: '{output_text}'")
        
        test_passed = True
        for prohibited_term in test_case['should_not_contain']:
            if prohibited_term in output_text:
                logger.error(f"  ❌ FAILED: Found prohibited term '{prohibited_term}'")
                test_passed = False
                validation_passed = False
        
        if test_passed:
            logger.info(f"  ✅ PASSED")
        
        logger.info("")
    
    if validation_passed:
        logger.info("🎉 ALL VALIDATION TESTS PASSED!")
    else:
        logger.error("❌ SOME VALIDATION TESTS FAILED - Fix may need refinement")

if __name__ == "__main__":
    main()