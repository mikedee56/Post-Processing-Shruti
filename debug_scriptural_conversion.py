#!/usr/bin/env python3
"""
Diagnostic test for Story 5.2 - Scriptural Conversion Failure

CRITICAL QA FINDING: "chapter two verse twenty five" → "Chapter 2 verse 25" NOT WORKING

This script isolates the scriptural conversion logic to identify the exact failure point.
"""

import sys
import logging
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Suppress verbose logging
logging.getLogger().setLevel(logging.ERROR)

def test_scriptural_conversion():
    """Test scriptural conversion in isolation."""
    print("=== DIAGNOSTIC: Scriptural Conversion Failure Analysis ===")
    print()
    
    # Test input from QA failure
    test_input = "chapter two verse twenty five"
    expected_output = "Chapter 2 verse 25"
    
    print(f"INPUT:    {test_input}")
    print(f"EXPECTED: {expected_output}")
    print()
    
    # Step 1: Test AdvancedTextNormalizer initialization
    try:
        from utils.advanced_text_normalizer import AdvancedTextNormalizer
        
        # Initialize with MCP disabled for focused testing
        config = {
            'enable_mcp_processing': False,
            'enable_fallback': True,
            'performance_optimized': True  # Disable complex integrations
        }
        normalizer = AdvancedTextNormalizer(config)
        print("✓ AdvancedTextNormalizer initialization: SUCCESS")
        
        # Test basic_numbers inheritance
        if hasattr(normalizer, 'basic_numbers'):
            print(f"✓ basic_numbers inheritance: SUCCESS ({len(normalizer.basic_numbers)} mappings)")
            print(f"  Sample: 'two' -> '{normalizer.basic_numbers.get('two', 'MISSING')}'")
            print(f"  Sample: 'twenty' -> '{normalizer.basic_numbers.get('twenty', 'MISSING')}'")
            print(f"  Sample: 'five' -> '{normalizer.basic_numbers.get('five', 'MISSING')}'")
        else:
            print("✗ basic_numbers inheritance: FAILED")
            return False
            
    except Exception as e:
        print(f"✗ AdvancedTextNormalizer initialization: FAILED - {e}")
        return False
    
    print()
    
    # Step 2: Test _word_to_digit method directly
    print("=== Testing _word_to_digit method ===")
    test_cases = ["two", "twenty", "five", "twenty five"]
    for word in test_cases:
        try:
            result = normalizer._word_to_digit(word)
            print(f"  '{word}' -> '{result}'")
        except Exception as e:
            print(f"  '{word}' -> ERROR: {e}")
    print()
    
    # Step 3: Test _convert_scriptural_numbers method directly
    print("=== Testing _convert_scriptural_numbers method ===")
    try:
        scriptural_result = normalizer._convert_scriptural_numbers(test_input)
        print(f"SCRIPTURAL RESULT: '{scriptural_result}'")
        
        if scriptural_result == expected_output:
            print("✓ Scriptural conversion: SUCCESS")
        else:
            print("✗ Scriptural conversion: FAILED")
            print(f"  Expected: '{expected_output}'")
            print(f"  Got:      '{scriptural_result}'")
            
            # Debug regex pattern
            print()
            print("=== Debugging regex pattern ===")
            pattern = r'\b(Chapter|chapter)\s+((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen))?)\s+(verse|sutra)\s+((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen))?)\b'
            
            matches = list(re.finditer(pattern, test_input, re.IGNORECASE))
            print(f"Pattern matches found: {len(matches)}")
            for i, match in enumerate(matches):
                print(f"  Match {i+1}: Groups = {match.groups()}")
                
    except Exception as e:
        print(f"✗ Scriptural conversion: ERROR - {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Step 4: Test full convert_numbers_with_context method
    print("=== Testing convert_numbers_with_context method ===")
    try:
        full_result = normalizer.convert_numbers_with_context(test_input)
        print(f"FULL RESULT: '{full_result}'")
        
        if full_result == expected_output:
            print("✓ Full context conversion: SUCCESS")
            return True
        else:
            print("✗ Full context conversion: FAILED")
            return False
            
    except Exception as e:
        print(f"✗ Full context conversion: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_classification():
    """Test context classification for scriptural input."""
    print()
    print("=== Testing Context Classification ===")
    
    try:
        from utils.advanced_text_normalizer import AdvancedTextNormalizer
        config = {'enable_mcp_processing': False, 'performance_optimized': True}
        normalizer = AdvancedTextNormalizer(config)
        
        test_input = "chapter two verse twenty five"
        
        # Test context classification
        context_type, confidence, segments = normalizer._classify_number_context_enhanced(test_input)
        print(f"Context Type: {context_type}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Segments: {segments}")
        
        if str(context_type) == "NumberContextType.SCRIPTURAL":
            print("✓ Context classification: CORRECT")
        else:
            print(f"✗ Context classification: INCORRECT (got {context_type})")
            
    except Exception as e:
        print(f"✗ Context classification: ERROR - {e}")

if __name__ == "__main__":
    success = test_scriptural_conversion()
    test_context_classification()
    
    print()
    if success:
        print("🎉 DIAGNOSTIC RESULT: Scriptural conversion is WORKING")
    else:
        print("⚠️  DIAGNOSTIC RESULT: Scriptural conversion is BROKEN")
        print("   Manual investigation required")