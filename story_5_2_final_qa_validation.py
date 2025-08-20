#!/usr/bin/env python3
"""
Story 5.2 Final QA Validation
ASCII-safe comprehensive validation of Story 5.2 MCP integration remediation.
"""

import sys
import tempfile
import os
from pathlib import Path
sys.path.insert(0, 'src')
import logging

# Configure minimal logging
logging.basicConfig(level=logging.ERROR)

def main():
    """Run final QA validation for Story 5.2."""
    print("=== Story 5.2 Final QA Validation ===")
    print()
    
    test_results = []
    validation_failures = []
    
    # Test 1: MCP Client Validation
    print("Test 1: MCP Client Operational Status")
    try:
        from utils.mcp_client import create_mcp_client
        
        client = create_mcp_client()
        performance_stats = client.get_performance_stats()
        compliance_report = client.get_professional_compliance_report()
        
        if performance_stats and compliance_report:
            print("  PASS: MCP Client operational")
            print("  PASS: Professional standards active")
            test_results.append(("MCP Client", "PASS"))
        else:
            print("  FAIL: MCP Client validation failed")
            test_results.append(("MCP Client", "FAIL"))
            validation_failures.append("MCP Client not operational")
            
    except Exception as e:
        print(f"  ERROR: MCP Client - {e}")
        test_results.append(("MCP Client", "ERROR"))
        validation_failures.append(f"MCP Client error: {e}")
    
    print()
    
    # Test 2: Advanced Text Normalizer
    print("Test 2: Advanced Text Normalizer Validation")
    try:
        from utils.advanced_text_normalizer import AdvancedTextNormalizer
        
        config = {'enable_mcp_processing': True, 'enable_fallback': True}
        normalizer = AdvancedTextNormalizer(config)
        
        test_cases = [
            ('chapter two verse twenty five', 'Chapter 2 verse 25'),
            ('And one by one, he killed six of their children.', 'And one by one, he killed six of their children.'),
            ('Year two thousand five.', 'Year 2005.')
        ]
        
        standalone_passed = True
        for input_text, expected in test_cases:
            result = normalizer.convert_numbers_with_context(input_text)
            passed = result == expected
            status = "PASS" if passed else "FAIL"
            print(f"  {status}: {input_text[:30]}...")
            if not passed:
                standalone_passed = False
                validation_failures.append(f"Text normalization failed for: {input_text}")
        
        if standalone_passed:
            print("  PASS: Advanced Text Normalizer fully functional")
            test_results.append(("Text Normalizer", "PASS"))
        else:
            print("  FAIL: Advanced Text Normalizer has issues")
            test_results.append(("Text Normalizer", "FAIL"))
            
    except Exception as e:
        print(f"  ERROR: Text Normalizer - {e}")
        test_results.append(("Text Normalizer", "ERROR"))
        validation_failures.append(f"Text Normalizer error: {e}")
    
    print()
    
    # Test 3: End-to-End Integration
    print("Test 3: End-to-End Integration Validation")
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        test_content = """1
00:00:01,000 --> 00:00:05,000
today we study krishna in chapter two verse twenty five."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            input_path = f.name
        
        output_path = input_path.replace('.srt', '_processed.srt')
        
        processor = SanskritPostProcessor()
        metrics = processor.process_srt_file(Path(input_path), Path(output_path))
        
        integration_passed = True
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                result = f.read()
            
            # Key validations
            validations = [
                ('Chapter 2 verse 25' in result, 'Scriptural conversion'),
                ('Krishna' in result, 'Sanskrit capitalization')
            ]
            
            for passed, description in validations:
                status = "PASS" if passed else "FAIL"
                print(f"  {status}: {description}")
                if not passed:
                    integration_passed = False
                    validation_failures.append(f"Integration {description} failed")
            
            print(f"  Processing time: {metrics.processing_time:.3f}s")
            
        else:
            print("  FAIL: No output file generated")
            integration_passed = False
            validation_failures.append("End-to-end processing failed")
        
        if integration_passed:
            print("  PASS: End-to-end integration functional")
            test_results.append(("Integration", "PASS"))
        else:
            print("  FAIL: End-to-end integration has gaps")
            test_results.append(("Integration", "FAIL"))
        
        # Cleanup
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
            
    except Exception as e:
        print(f"  ERROR: Integration - {e}")
        test_results.append(("Integration", "ERROR"))
        validation_failures.append(f"Integration error: {e}")
    
    print()
    
    # Final Assessment
    print("=== Final Assessment ===")
    passed_tests = sum(1 for _, result in test_results if result == "PASS")
    total_tests = len(test_results)
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    for test_name, result in test_results:
        print(f"  {test_name}: {result}")
    
    print()
    
    if validation_failures:
        print("VALIDATION FAILURES:")
        for i, failure in enumerate(validation_failures, 1):
            print(f"  {i}. {failure}")
        print()
    
    # Overall Status
    if passed_tests == total_tests and not validation_failures:
        print("SUCCESS: Story 5.2 PRODUCTION READY")
        print("All validations passed - system ready for deployment")
        return True
    else:
        print("WARNING: Story 5.2 REQUIRES REMEDIATION")
        print(f"Issues detected: {len(validation_failures)}")
        print("Professional standards framework preventing compromised deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)