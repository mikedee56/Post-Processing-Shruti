#!/usr/bin/env python3
"""
Debug the corruption fix safely with no console Unicode issues
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Suppress logging to avoid Unicode issues
logging.getLogger().setLevel(logging.ERROR)

def safe_print(text):
    """Print text safely by handling Unicode characters."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters with safe representations
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

def test_corruption_fix():
    """Test if the corruption fix is actually being applied."""
    safe_print("=== CORRUPTION FIX DEBUG ===")
    safe_print("")
    
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    
    processor = SanskritPostProcessor()
    
    # Test direct text that we know gets corrupted
    test_text = "today we study krishna and dharma"
    
    safe_print(f"Original: {repr(test_text)}")
    
    # Test the enhanced Sanskrit/Hindi corrections method directly
    result = processor._apply_enhanced_sanskrit_hindi_corrections(test_text)
    corrected = result['corrected_text']
    
    safe_print(f"Corrected (safe): {corrected.encode('ascii', errors='replace').decode('ascii')}")
    
    # Check if corruption fix was applied by looking for the fixed terms
    if "Krishna" in corrected:
        safe_print("✅ Krishna successfully corrected/preserved")
    elif "K" in corrected and "?" in corrected:
        safe_print("❌ Krishna corruption still present")
        safe_print("DETAILED: Checking each character:")
        for i, char in enumerate(corrected):
            if ord(char) > 127:  # Non-ASCII character
                safe_print(f"  Position {i}: '{char}' (ord={ord(char)})")
    else:
        safe_print("⚠️ Krishna not found in result")
    
    # Test NER capitalization directly
    safe_print("\n=== Testing NER Capitalization ===")
    if processor.enable_ner and processor.capitalization_engine:
        cap_result = processor.capitalization_engine.capitalize_text("today we study krishna")
        safe_print(f"NER result: {cap_result.capitalized_text.encode('ascii', errors='replace').decode('ascii')}")
        safe_print(f"Changes made: {len(cap_result.changes_made)}")
        
        if "Krishna" in cap_result.capitalized_text:
            safe_print("✅ NER Krishna capitalization working")
        else:
            safe_print("❌ NER Krishna capitalization failing")

if __name__ == "__main__":
    test_corruption_fix()