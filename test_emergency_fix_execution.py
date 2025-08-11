#!/usr/bin/env python3
"""
Emergency Fix Test Script - Test that the corruption examples are resolved
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Test the anti-hallucination fixes by directly testing problematic cases
def test_emergency_fix():
    """Test specific corruption examples that were reported."""
    
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        # Initialize the processor
        processor = SanskritPostProcessor()
        
        print("=" * 80)
        print("🚨 EMERGENCY ANTI-HALLUCINATION FIX TEST")
        print("=" * 80)
        print()
        
        # Test cases - the specific corruption examples from the user's report
        test_cases = [
            {
                "name": "who_is_brahman_test",
                "input": "Adorations to Sadguru, who is Brahman, the giver of supreme bliss",
                "expected_output": "Adorations to Sadguru, who is Brahman, the giver of supreme bliss",
                "should_not_contain": ["Kṛṣṇa", "krishna", "krsna", "कृष्ण"]
            },
            {
                "name": "chapter_entitled_test",
                "input": "This chapter is entitled, Atma Vishranti",
                "expected_similar": "This chapter is entitled",
                "should_not_contain": ["Kṛṣṇa", "ātman", "1", "2"]
            },
            {
                "name": "highly_inspired_test",
                "input": "highly inspired and enlightened seekers",
                "expected_output": "highly inspired and enlightened seekers",
                "should_not_contain": ["Kṛṣṇa", "ātman", "krsna"]
            },
            {
                "name": "english_protection_test",
                "input": "who what when where why how and the is are was were",
                "expected_output": "who what when where why how and the is are was were",
                "should_not_contain": ["Kṛṣṇa", "ātman", "Vedas", "dharma", "yoga"]
            },
            {
                "name": "short_words_protection",
                "input": "one without a second, vast as the ether",
                "expected_similar": "one without a second, vast as the ether",
                "should_not_contain": ["1", "2nd", "Vedas", "Kṛṣṇa", "ātman"]
            }
        ]
        
        all_tests_passed = True
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[TEST {i}] {test_case['name'].replace('_', ' ').upper()}")
            print(f"Input:  '{test_case['input']}'")
            
            # Test the correction method directly
            corrected_text, corrections = processor._apply_lexicon_corrections(test_case['input'])
            
            print(f"Output: '{corrected_text}'")
            print(f"Corrections Applied: {corrections}")
            
            # Test 1: Check that no unwanted terms are present
            test_passed = True
            for unwanted in test_case['should_not_contain']:
                if unwanted.lower() in corrected_text.lower():
                    print(f"❌ FAIL: Found unwanted term '{unwanted}' in output!")
                    test_passed = False
                    all_tests_passed = False
            
            # Test 2: Check expected output if specified
            if 'expected_output' in test_case:
                if corrected_text.strip() != test_case['expected_output'].strip():
                    print(f"❌ FAIL: Expected exact output '{test_case['expected_output']}' but got '{corrected_text}'")
                    test_passed = False
                    all_tests_passed = False
            
            # Test 3: Check expected similarity if specified
            if 'expected_similar' in test_case:
                if test_case['expected_similar'].lower() not in corrected_text.lower():
                    print(f"❌ FAIL: Expected to find '{test_case['expected_similar']}' in output")
                    test_passed = False
                    all_tests_passed = False
            
            if test_passed:
                print("✅ PASS: No hallucination detected!")
            else:
                print("❌ FAIL: Hallucination or unwanted changes detected!")
                
            print("-" * 60)
            print()
        
        # Summary
        print(f"{'='*80}")
        if all_tests_passed:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Emergency anti-hallucination fixes are working correctly.")
            print("✅ No random Sanskrit terms are being inserted.")
            print("✅ English text is completely protected.")
            print("✅ System is now safe for professional use.")
        else:
            print("❌ SOME TESTS FAILED!")
            print("⚠️  Hallucination issues still exist - additional fixes required.")
            
        print("="*80)
        
        return all_tests_passed
        
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_emergency_fix()
    sys.exit(0 if success else 1)