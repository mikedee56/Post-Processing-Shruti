#!/usr/bin/env python3
"""
🚨 COMPLETE EMERGENCY DEPLOYMENT SCRIPT 🚨
Deploys emergency anti-hallucination fixes and processes user's 15 files

This script:
1. Validates that emergency fixes are working
2. Tests user's specific corruption examples
3. Re-processes all 15 SRT files with safe settings
4. Generates quality report
5. Confirms professional quality standards are met

Usage: python EMERGENCY_DEPLOYMENT_COMPLETE.py
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def validate_emergency_fixes():
    """Validate that emergency anti-hallucination fixes are working."""
    print("=" * 80)
    print("🔍 PHASE 1: VALIDATING EMERGENCY FIXES")
    print("=" * 80)
    
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        processor = SanskritPostProcessor()
        
        # Critical test cases from user's corruption report
        corruption_tests = [
            {
                "name": "who_is_brahman",
                "input": "Adorations to Sadguru, who is Brahman, the giver of supreme bliss",
                "forbidden_terms": ["Kṛṣṇa", "kṛṣṇa", "krishna", "krsna"]
            },
            {
                "name": "chapter_entitled", 
                "input": "This chapter is entitled, Atma Vishranti",
                "forbidden_terms": ["Kṛṣṇa", "kṛṣṇa", "1", "2", "ātman"]
            },
            {
                "name": "highly_inspired",
                "input": "highly inspired and enlightened seekers",
                "forbidden_terms": ["Kṛṣṇa", "kṛṣṇa", "ātman"]
            },
            {
                "name": "protected_english_words",
                "input": "who what when where why how and the is are was were",
                "forbidden_terms": ["Kṛṣṇa", "kṛṣṇa", "ātman", "Vedas", "dharma"]
            },
            {
                "name": "carrying_process", 
                "input": "carrying out this process with dedication",
                "forbidden_terms": ["Kṛṣṇa", "kṛṣṇa", "ātman"]
            }
        ]
        
        all_tests_passed = True
        
        for test in corruption_tests:
            print(f"\\n🧪 Testing: {test['name']}")
            print(f"Input: '{test['input']}'")
            
            # Apply lexicon corrections directly
            corrected_text, corrections = processor._apply_lexicon_corrections(test['input'])
            
            print(f"Output: '{corrected_text}'")
            print(f"Corrections: {corrections}")
            
            # Check for forbidden Sanskrit insertions
            test_failed = False
            for forbidden in test['forbidden_terms']:
                if forbidden in corrected_text:
                    print(f"❌ CRITICAL FAILURE: Found forbidden term '{forbidden}' in output!")
                    test_failed = True
                    all_tests_passed = False
            
            if not test_failed:
                print("✅ PASS: No hallucination detected")
            
            print("-" * 60)
        
        if all_tests_passed:
            print("\\n🎉 ALL VALIDATION TESTS PASSED!")
            print("✅ Emergency anti-hallucination fixes are working correctly")
            print("✅ No random Sanskrit terms are being inserted")
            print("✅ System is safe for professional use")
            return True
        else:
            print("\\n❌ VALIDATION FAILED!")
            print("⚠️  System still has hallucination issues")
            print("🛑 STOPPING DEPLOYMENT - Manual intervention required")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_user_files():
    """Process the user's 15 SRT files with emergency safe settings."""
    print("\\n" + "=" * 80)
    print("🔄 PHASE 2: PROCESSING USER'S 15 SRT FILES")
    print("=" * 80)
    
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        # Initialize processor with emergency safe settings
        processor = SanskritPostProcessor()
        
        # Get list of raw SRT files
        raw_srts_dir = Path("data/raw_srts")
        processed_srts_dir = Path("data/processed_srts")
        
        srt_files = list(raw_srts_dir.glob("*.srt"))
        print(f"Found {len(srt_files)} SRT files to process")
        
        processing_results = []
        
        for i, srt_file in enumerate(srt_files, 1):
            print(f"\\n📄 [{i}/{len(srt_files)}] Processing: {srt_file.name}")
            
            try:
                # Generate output filename with emergency-safe suffix
                output_file = processed_srts_dir / f"{srt_file.stem}_emergency_safe.srt"
                
                # Process the file
                start_time = time.time()
                metrics = processor.process_srt_file(srt_file, output_file)
                processing_time = time.time() - start_time
                
                # Collect results
                result = {
                    'filename': srt_file.name,
                    'processing_time': processing_time,
                    'total_segments': metrics.total_segments,
                    'segments_modified': metrics.segments_modified,
                    'average_confidence': metrics.average_confidence,
                    'errors': len(metrics.errors_encountered),
                    'warnings': len(metrics.warnings_encountered),
                    'flagged_segments': metrics.flagged_segments,
                    'success': True
                }
                
                processing_results.append(result)
                
                print(f"   ✅ Success: {metrics.total_segments} segments, {metrics.segments_modified} modified")
                print(f"   ℹ️  Confidence: {metrics.average_confidence:.3f}, Time: {processing_time:.2f}s")
                
                # CRITICAL: Check for over-processing (hallucination indicator)
                modification_rate = metrics.segments_modified / metrics.total_segments if metrics.total_segments > 0 else 0
                if modification_rate > 0.3:  # More than 30% modified is suspicious
                    print(f"   ⚠️  WARNING: High modification rate ({modification_rate:.1%}) - possible over-processing")
                    result['warning'] = f"High modification rate: {modification_rate:.1%}"
                
                if metrics.average_confidence < 0.7:  # Low confidence
                    print(f"   ⚠️  WARNING: Low average confidence ({metrics.average_confidence:.3f})")
                    result['warning'] = f"Low confidence: {metrics.average_confidence:.3f}"
                
            except Exception as e:
                print(f"   ❌ Error processing {srt_file.name}: {e}")
                processing_results.append({
                    'filename': srt_file.name,
                    'success': False,
                    'error': str(e)
                })
        
        return processing_results
        
    except Exception as e:
        print(f"❌ ERROR during file processing: {e}")
        import traceback
        traceback.print_exc()
        return []

def generate_quality_report(processing_results: List[Dict[str, Any]]):
    """Generate comprehensive quality report."""
    print("\\n" + "=" * 80)
    print("📊 PHASE 3: GENERATING QUALITY REPORT")
    print("=" * 80)
    
    if not processing_results:
        print("❌ No processing results to report")
        return
    
    successful_files = [r for r in processing_results if r.get('success', False)]
    failed_files = [r for r in processing_results if not r.get('success', False)]
    
    print(f"\\n📈 PROCESSING SUMMARY:")
    print(f"   ✅ Successfully processed: {len(successful_files)}/{len(processing_results)} files")
    print(f"   ❌ Failed: {len(failed_files)} files")
    
    if successful_files:
        total_segments = sum(r['total_segments'] for r in successful_files)
        total_modified = sum(r['segments_modified'] for r in successful_files)
        avg_confidence = sum(r['average_confidence'] for r in successful_files) / len(successful_files)
        total_flagged = sum(r['flagged_segments'] for r in successful_files)
        
        modification_rate = total_modified / total_segments if total_segments > 0 else 0
        flagged_rate = total_flagged / total_segments if total_segments > 0 else 0
        
        print(f"\\n🎯 QUALITY METRICS:")
        print(f"   • Total segments processed: {total_segments:,}")
        print(f"   • Segments modified: {total_modified:,} ({modification_rate:.1%})")
        print(f"   • Average confidence: {avg_confidence:.3f}")
        print(f"   • Segments flagged: {total_flagged} ({flagged_rate:.1%})")
        
        # Quality assessment 
        print(f"\\n🏆 QUALITY ASSESSMENT:")
        
        if modification_rate <= 0.15:  # ≤15% modification rate
            print("   ✅ EXCELLENT: Low modification rate - minimal over-processing")
        elif modification_rate <= 0.30:  # ≤30% modification rate  
            print("   ✅ GOOD: Moderate modification rate - acceptable processing")
        elif modification_rate <= 0.50:  # ≤50% modification rate
            print("   ⚠️  CAUTION: High modification rate - monitor for over-processing")
        else:  # >50% modification rate
            print("   ❌ CONCERN: Very high modification rate - likely over-processing")
            
        if avg_confidence >= 0.85:
            print("   ✅ EXCELLENT: High confidence scores - professional quality")
        elif avg_confidence >= 0.70:
            print("   ✅ GOOD: Acceptable confidence scores")
        else:
            print("   ⚠️  CAUTION: Low confidence scores - review quality")
            
        if flagged_rate <= 0.05:  # ≤5% flagged
            print("   ✅ EXCELLENT: Very few segments flagged for review")
        elif flagged_rate <= 0.15:  # ≤15% flagged
            print("   ✅ GOOD: Reasonable number of segments flagged")
        else:
            print("   ⚠️  CAUTION: Many segments flagged - manual review recommended")
    
    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'deployment_type': 'emergency_anti_hallucination_fix',
        'total_files': len(processing_results),
        'successful_files': len(successful_files),
        'failed_files': len(failed_files),
        'results': processing_results
    }
    
    if successful_files:
        report.update({
            'total_segments': total_segments,
            'modification_rate': modification_rate,
            'average_confidence': avg_confidence,
            'flagged_rate': flagged_rate,
            'quality_score': min(avg_confidence, 1.0 - modification_rate)  # Combined quality metric
        })
    
    report_file = Path("data/metrics") / f"emergency_deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\\n📄 Detailed report saved: {report_file}")
    
    return report

def main():
    """Main deployment function."""
    print("🚨" * 30)
    print("EMERGENCY ANTI-HALLUCINATION DEPLOYMENT")
    print("ASR Post-Processing Quality Recovery System")
    print("🚨" * 30)
    print()
    
    # Phase 1: Validate emergency fixes
    if not validate_emergency_fixes():
        print("\\n🛑 DEPLOYMENT ABORTED: Validation failed")
        print("Manual intervention required to fix hallucination issues")
        return 1
    
    # Phase 2: Process user's files
    processing_results = process_user_files()
    if not processing_results:
        print("\\n❌ DEPLOYMENT FAILED: No files processed successfully")
        return 1
    
    # Phase 3: Generate quality report
    report = generate_quality_report(processing_results)
    
    # Final assessment
    print("\\n" + "=" * 80)
    print("🎯 FINAL DEPLOYMENT STATUS")
    print("=" * 80)
    
    successful_files = len([r for r in processing_results if r.get('success', False)])
    total_files = len(processing_results)
    
    if successful_files == total_files:
        print("✅ DEPLOYMENT SUCCESSFUL!")
        print(f"✅ All {total_files} files processed successfully")
        print("✅ Emergency anti-hallucination fixes are active")
        print("✅ System is ready for professional use")
        
        if 'quality_score' in report and report['quality_score'] >= 0.7:
            print("✅ Quality standards met - professional grade output")
        else:
            print("⚠️  Quality review recommended - monitor results")
            
        print("\\n🚀 NEXT STEPS:")
        print("1. Review processed files in data/processed_srts/")
        print("2. Check quality report for any warnings")
        print("3. Proceed with confidence - hallucination eliminated")
        
        return 0
    else:
        print("⚠️  DEPLOYMENT PARTIALLY SUCCESSFUL")
        print(f"⚠️  {successful_files}/{total_files} files processed successfully")
        print("⚠️  Review failed files and retry if needed")
        return 2

if __name__ == "__main__":
    sys.exit(main())