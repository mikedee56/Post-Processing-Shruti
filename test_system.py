#!/usr/bin/env python3
"""
Simple system test for Epic 2.4 - Windows 11 Setup Verification
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from post_processors.sanskrit_post_processor import SanskritPostProcessor
from utils.srt_parser import SRTParser

def main():
    print("=== Epic 2.4 System Test ===")
    print()
    
    # Test 1: Component initialization
    print("Test 1: Component Initialization")
    try:
        processor = SanskritPostProcessor()
        parser = SRTParser()
        print("SUCCESS: All components initialized successfully")
    except Exception as e:
        print(f"FAILED: Initialization failed: {e}")
        return False
    
    print()
    
    # Test 2: File processing
    print("Test 2: SRT File Processing")
    
    # Check available test files
    test_files = [
        Path("data/test_samples/basic_test.srt"),
        Path("data/raw_srts/WhisperX lg v2.srt"),
        Path("data/test_samples/conversational_test.srt")
    ]
    
    processed_any = False
    for test_file in test_files:
        if test_file.exists():
            try:
                output_file = Path("data/processed_srts") / f"{test_file.stem}_test_enhanced.srt"
                print(f"  Processing: {test_file}")
                
                metrics = processor.process_srt_file(test_file, output_file)
                
                print(f"  SUCCESS: {metrics.total_segments} segments, {metrics.segments_modified} enhanced")
                print(f"    Confidence: {metrics.average_confidence:.3f}, Time: {metrics.processing_time:.2f}s")
                processed_any = True
                break
                
            except Exception as e:
                print(f"  FAILED to process {test_file}: {e}")
                continue
    
    if not processed_any:
        print("  FAILED: No test files could be processed")
        return False
    
    print()
    print("Epic 2.4 System Test: PASSED")
    print("Your system is ready for production use!")
    print()
    print("Next Steps:")
    print("1. Place your .srt files in data/raw_srts/")
    print("2. Run: py -3.10 test_system.py (for testing)")
    print("3. For batch processing: py -3.10 -c \"exec(open('simple_batch.py').read())\"")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)