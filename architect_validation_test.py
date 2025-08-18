#!/usr/bin/env python3
"""
Architect Validation Test - Independent QA Validation
======================================================

This script provides independent validation of the critical fixes implemented
by the architect, using the novel QA framework approach.
"""

import sys
sys.path.insert(0, 'src')

from utils.advanced_text_normalizer import AdvancedTextNormalizer
from post_processors.sanskrit_post_processor import SanskritPostProcessor
import logging

# Suppress verbose logging for clean test output
logging.getLogger().setLevel(logging.ERROR)

def test_critical_fixes():
    """Test the three critical functionality failures identified by QA."""
    
    print("=" * 60)
    print("ARCHITECT VALIDATION TEST")
    print("Independent QA Validation of Critical Fixes")
    print("=" * 60)
    print()
    
    # Initialize the text normalizer
    config = {'enable_mcp_processing': True, 'enable_fallback': True}
    normalizer = AdvancedTextNormalizer(config)
    
    # Test cases from QA audit
    critical_test_cases = [
        {
            'name': 'Scriptural Reference Conversion',
            'input': 'chapter two verse twenty five',
            'expected': 'Chapter 2 verse 25',
            'description': 'Convert scriptural references with proper capitalization'
        },
        {
            'name': 'Idiomatic Expression Preservation', 
            'input': 'And one by one, he killed six of their children.',
            'expected': 'And one by one, he killed six of their children.',
            'description': 'Preserve idiomatic expressions unchanged'
        },
        {
            'name': 'Temporal Number Conversion',
            'input': 'Year two thousand five.',
            'expected': 'Year 2005.',
            'description': 'Convert temporal/year references correctly'
        }
    ]
    
    # Run validation tests
    all_tests_passed = True
    test_results = []
    
    for i, test_case in enumerate(critical_test_cases, 1):
        print(f"TEST {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Input:    '{test_case['input']}'")
        print(f"Expected: '{test_case['expected']}'")
        
        # Run the test
        try:
            result = normalizer.convert_numbers_with_context(test_case['input'])
            passed = result == test_case['expected']
            
            print(f"Actual:   '{result}'")
            print(f"Status:   {'PASS' if passed else 'FAIL'}")
            
            test_results.append({
                'test': test_case['name'],
                'passed': passed,
                'input': test_case['input'],
                'expected': test_case['expected'],
                'actual': result
            })
            
            if not passed:
                all_tests_passed = False
                
        except Exception as e:
            print(f"Actual:   ERROR - {e}")
            print(f"Status:   FAIL (Exception)")
            all_tests_passed = False
            test_results.append({
                'test': test_case['name'],
                'passed': False,
                'error': str(e)
            })
        
        print("-" * 40)
        print()
    
    return all_tests_passed, test_results

def test_integration_functionality():
    """Test end-to-end integration with Sanskrit post-processor."""
    
    print("INTEGRATION TEST: Sanskrit Post-Processor")
    print("Testing combined text normalization and capitalization")
    print("-" * 40)
    
    try:
        # Initialize post-processor
        processor = SanskritPostProcessor()
        
        # Test text that needs both normalization and capitalization
        test_text = "today we study krishna in chapter two verse twenty five"
        
        print(f"Input: '{test_text}'")
        
        # Test text normalization
        normalized = processor.text_normalizer.convert_numbers_with_context(test_text)
        print(f"After text normalization: '{normalized}'")
        
        # Test capitalization (if available)
        if processor.capitalization_engine:
            cap_result = processor.capitalization_engine.capitalize_text(normalized)
            capitalized = cap_result.capitalized_text
            print(f"After capitalization: '{capitalized}'")
            
            # Validate both transformations occurred
            # Note: After capitalization, "verse" becomes "Verse"  
            has_normalization = ("Chapter 2 verse 25" in capitalized or "Chapter 2 Verse 25" in capitalized)
            has_capitalization = "Krishna" in capitalized
            
            print()
            print(f"Text normalization applied: {'YES' if has_normalization else 'NO'}")
            print(f"Capitalization applied: {'YES' if has_capitalization else 'NO'}")
            
            integration_success = has_normalization and has_capitalization
            print(f"Integration success: {'PASS' if integration_success else 'FAIL'}")
            
            return integration_success
        else:
            print("FAIL: Capitalization engine not available")
            return False
            
    except Exception as e:
        print(f"FAIL: Integration test error - {e}")
        return False

def generate_validation_report(test_results, integration_success):
    """Generate final validation report."""
    
    print()
    print("=" * 60)
    print("FINAL VALIDATION REPORT")
    print("=" * 60)
    print()
    
    # Summary statistics
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result.get('passed', False))
    
    print(f"Critical Functionality Tests: {passed_tests}/{total_tests} PASSED")
    print(f"Integration Test: {'PASS' if integration_success else 'FAIL'}")
    print()
    
    # Detailed results
    print("DETAILED RESULTS:")
    for result in test_results:
        status = "PASS" if result.get('passed', False) else "FAIL"
        print(f"  {result['test']}: {status}")
        if not result.get('passed', False) and 'error' not in result:
            print(f"    Expected: {result.get('expected', 'N/A')}")
            print(f"    Actual:   {result.get('actual', 'N/A')}")
    
    print()
    
    # Overall assessment
    overall_success = (passed_tests == total_tests) and integration_success
    
    if overall_success:
        print("OVERALL ASSESSMENT: ALL CRITICAL FIXES SUCCESSFUL")
        print("System functionality restored to claimed standards")
        print("Ready for production deployment")
    else:
        print("OVERALL ASSESSMENT: CRITICAL ISSUES REMAIN")
        print("Additional fixes required before production deployment")
    
    print()
    print("=" * 60)
    
    return overall_success

def main():
    """Main validation routine."""
    
    # Run critical functionality tests
    critical_success, test_results = test_critical_fixes()
    
    # Run integration tests
    integration_success = test_integration_functionality()
    
    # Generate final report
    overall_success = generate_validation_report(test_results, integration_success)
    
    # Return appropriate exit code
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    main()