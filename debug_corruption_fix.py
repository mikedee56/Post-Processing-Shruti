#!/usr/bin/env python3
"""
Debug the corruption fix to see if it's being applied
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Enable logging to see corruption fixes
logging.basicConfig(level=logging.INFO)

def test_corruption_fix():
    """Test if the corruption fix is actually being applied."""
    print("=== CORRUPTION FIX DEBUG ===")
    print()
    
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    
    processor = SanskritPostProcessor()
    
    # Test direct text that we know gets corrupted
    test_text = "today we study krishna and dharma"
    
    print(f"Original: {repr(test_text)}")
    
    # Test the enhanced Sanskrit/Hindi corrections method directly
    result = processor._apply_enhanced_sanskrit_hindi_corrections(test_text)
    corrected = result['corrected_text']
    
    print(f"Corrected: {repr(corrected)}")
    
    # Check if corruption fix was applied by looking for the fixed terms
    if "Krishna" in corrected:
        print("✅ Krishna successfully corrected/preserved")
    elif "K" in corrected and "?" in corrected:
        print("❌ Krishna corruption still present")
    else:
        print("⚠️ Krishna not found in result")
    
    # Test with already corrupted text
    print("\n=== Testing with pre-corrupted text ===")
    corrupted_text = "today we study K???a and Vi??u"
    print(f"Corrupted input: {repr(corrupted_text)}")
    
    result2 = processor._apply_enhanced_sanskrit_hindi_corrections(corrupted_text)
    fixed = result2['corrected_text']
    
    print(f"Fixed result: {repr(fixed)}")
    
    if "Krishna" in fixed and "Vishnu" in fixed:
        print("✅ Corruption fix working correctly")
    else:
        print("❌ Corruption fix not working")

if __name__ == "__main__":
    test_corruption_fix()