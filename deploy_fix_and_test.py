#!/usr/bin/env python3
"""
Deployment Script - Test Anti-Hallucination Fixes on Real Data
This script processes a sample SRT file to verify the fixes work in production.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from post_processors.sanskrit_post_processor import SanskritPostProcessor

def deploy_and_test():
    """Deploy the fixes and test on actual data."""
    
    print("="*80)
    print("DEPLOYMENT TEST - ANTI-HALLUCINATION FIXES")
    print("="*80)
    
    # Initialize processor
    try:
        processor = SanskritPostProcessor()
        print("✅ SanskritPostProcessor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        return False
    
    # Test input and output paths
    test_input = Path("data/raw_srts/Whisperx lg v2.srt")
    test_output = Path("data/processed_srts/Whisperx lg v2_FIXED.srt")
    
    if not test_input.exists():
        print(f"❌ Test input file not found: {test_input}")
        return False
    
    print(f"📄 Input file: {test_input}")
    print(f"📄 Output file: {test_output}")
    
    # Process the file
    try:
        print("\n🔄 Processing file with anti-hallucination fixes...")
        metrics = processor.process_srt_file(test_input, test_output)
        
        print(f"✅ Processing completed successfully!")
        print(f"📊 Segments processed: {metrics.total_segments}")
        print(f"📊 Segments modified: {metrics.segments_modified}")
        print(f"📊 Average confidence: {metrics.average_confidence:.3f}")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify the specific corruption examples are fixed
    print("\n🔍 Verifying anti-hallucination fixes...")
    
    try:
        with open(test_output, 'r', encoding='utf-8') as f:
            processed_content = f.read()
        
        # Check for the specific corruption patterns that were reported
        corruption_patterns = [
            ("who Kṛṣṇa Brahman", "should be 'who is Brahman'"),
            ("chapter 1 Kṛṣṇa entitled", "should be 'chapter is entitled'"),
            ("1 without a 2nd", "should be 'one without a second'"),
            ("vast Vedas the ether", "should be 'vast as the ether'")
        ]
        
        corruptions_found = []
        for pattern, description in corruption_patterns:
            if pattern in processed_content:
                corruptions_found.append(f"❌ Found: {pattern} ({description})")
        
        if corruptions_found:
            print("⚠️  CORRUPTION STILL DETECTED:")
            for corruption in corruptions_found:
                print(f"  {corruption}")
            return False
        else:
            print("✅ No corruption patterns detected in output!")
            
        # Check that legitimate Sanskrit terms are still processed
        legitimate_patterns = [
            "Bhagavad Gītā",
            "Vedānta", 
            "dharma",
            "yoga"
        ]
        
        legitimate_found = []
        for pattern in legitimate_patterns:
            if pattern in processed_content:
                legitimate_found.append(pattern)
        
        if legitimate_found:
            print(f"✅ Legitimate Sanskrit terms preserved: {', '.join(legitimate_found)}")
        else:
            print("ℹ️  No obvious legitimate Sanskrit terms found (this may be normal for this file)")
            
    except Exception as e:
        print(f"❌ Error reading output file: {e}")
        return False
    
    print("\n" + "="*80)
    print("🎉 DEPLOYMENT TEST SUCCESSFUL!")
    print("✅ Anti-hallucination fixes are working correctly")
    print("✅ File processed without corruption")  
    print("✅ Ready for production deployment")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = deploy_and_test()
    sys.exit(0 if success else 1)