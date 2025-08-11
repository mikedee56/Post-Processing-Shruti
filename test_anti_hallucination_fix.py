#!/usr/bin/env python3
"""
Test Script to Verify Anti-Hallucination Fixes
This script tests specific corruption examples to ensure they are eliminated.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from post_processors.sanskrit_post_processor import SanskritPostProcessor
import tempfile

def test_anti_hallucination():
    """Test that the specific corruption examples are fixed."""
    
    # Initialize the processor with anti-hallucination config
    processor = SanskritPostProcessor()
    
    # Test cases - examples from the user's corruption report
    test_cases = [
        {
            "name": "who_is_brahman",
            "input": "Adorations to Sadguru, who is Brahman, the giver of supreme bliss",
            "should_not_contain": ["Kṛṣṇa", "krishna", "krsna"],
            "should_contain": "who is Brahman"
        },
        {
            "name": "chapter_entitled", 
            "input": "This chapter is entitled, Atma Vishranti",
            "should_not_contain": ["Kṛṣṇa", "ātman", "1", "2"],
            "should_contain": "This chapter is entitled"
        },
        {
            "name": "highly_inspired",
            "input": "highly inspired and enlightened",
            "should_not_contain": ["Kṛṣṇa", "ātman"],
            "should_contain": ["highly inspired", "and"]
        },
        {
            "name": "one_without_second",
            "input": "one without a second, vast as the ether",
            "should_not_contain": ["1", "2nd", "Vedas"],
            "should_contain": ["one without a second", "vast as the ether"]
        },
        {
            "name": "protected_words",
            "input": "who what when where why how and the is are was were",
            "should_not_contain": ["Kṛṣṇa", "ātman", "Vedas", "dharma"],
            "should_contain": "who what when where why how and the is are was were"
        }
    ]
    
    print("="*80)
    print("ANTI-HALLUCINATION FIX TEST")
    print("="*80)
    
    all_tests_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {test_case['name'].upper()}")
        print(f"Input: {test_case['input']}")
        
        # Test the legacy correction method directly
        corrected_text, corrections = processor._apply_lexicon_corrections(test_case['input'])
        
        print(f"Output: {corrected_text}")
        print(f"Corrections Applied: {corrections}")
        
        # Check that unwanted terms are NOT present
        test_passed = True
        for unwanted in test_case['should_not_contain']:
            if unwanted.lower() in corrected_text.lower():
                print(f"❌ FAIL: Found unwanted term '{unwanted}' in output!")
                test_passed = False
                all_tests_passed = False
        
        # Check that required terms ARE present
        for required in test_case.get('should_contain', []):
            if isinstance(required, str):
                if required.lower() not in corrected_text.lower():
                    print(f"❌ FAIL: Required term '{required}' missing from output!")
                    test_passed = False
                    all_tests_passed = False
            elif isinstance(required, list):
                for req in required:
                    if req.lower() not in corrected_text.lower():
                        print(f"❌ FAIL: Required term '{req}' missing from output!")
                        test_passed = False
                        all_tests_passed = False
        
        if test_passed:
            print("✅ PASS: No hallucination detected!")
        
        print("-" * 60)
    
    print(f"\n{'='*80}")
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Anti-hallucination fixes are working correctly.")
        print("✅ The system will no longer inject random Sanskrit terms.")
        print("✅ Protected English words are now completely safe.")
    else:
        print("❌ SOME TESTS FAILED! Hallucination issues still exist.")
        print("⚠️  Additional fixes may be required.")
    print("="*80)
    
    return all_tests_passed

def test_fuzzy_matcher_directly():
    """Test the fuzzy matcher component directly."""
    print("\n" + "="*60)
    print("TESTING FUZZY MATCHER DIRECTLY")
    print("="*60)
    
    processor = SanskritPostProcessor()
    fuzzy_matcher = processor.fuzzy_matcher
    
    # Test protected words that should NEVER match
    protected_test_words = ['who', 'is', 'and', 'the', 'as', 'one', 'two', 'chapter', 'entitled']
    
    for word in protected_test_words:
        matches = fuzzy_matcher.find_matches(word)
        print(f"Testing '{word}': {len(matches)} matches found")
        
        if matches:
            print(f"❌ WARNING: Protected word '{word}' has {len(matches)} matches!")
            for match in matches:
                print(f"  - Match: {match.original_word} -> {match.transliteration} (confidence: {match.confidence})")
        else:
            print(f"✅ Good: Protected word '{word}' has no matches")
    
    print("=" * 60)

if __name__ == "__main__":
    # Run the tests
    success = test_anti_hallucination()
    test_fuzzy_matcher_directly()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)