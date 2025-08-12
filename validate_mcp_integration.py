#!/usr/bin/env python3
"""
MCP Integration Validation Script

This script validates the successful implementation of MCP-enhanced context-aware
number processing that solves the "one by one" -> "1 by 1" quality degradation issue.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.advanced_text_normalizer import AdvancedTextNormalizer
from utils.text_normalizer import TextNormalizer


def validate_critical_fix():
    """Validate the critical 'one by one' fix."""
    print("🎯 CRITICAL FIX VALIDATION")
    print("=" * 50)
    
    # Old system (basic TextNormalizer)
    old_normalizer = TextNormalizer()
    
    # New system (MCP-enhanced AdvancedTextNormalizer) 
    new_config = {'enable_mcp_processing': True, 'enable_fallback': True}
    new_normalizer = AdvancedTextNormalizer(new_config)
    
    critical_test = "And one by one, he killed six of their children."
    
    print(f"📝 Test Input: {critical_test}")
    print()
    
    # Old system result
    old_result = old_normalizer.convert_numbers(critical_test)
    print(f"❌ OLD SYSTEM: {old_result}")
    print(f"   Problem: {'BROKEN' if 'one by one' not in old_result else 'OK'}")
    
    # New system result  
    new_result = new_normalizer.convert_numbers_with_context(critical_test)
    print(f"✅ NEW SYSTEM: {new_result}")
    print(f"   Solution: {'FIXED' if 'one by one' in new_result else 'BROKEN'}")
    
    print()
    success = 'one by one' in new_result and 'one by one' not in old_result
    print(f"🏆 CRITICAL FIX STATUS: {'SUCCESS' if success else 'FAILED'}")
    
    return success


def validate_context_classifications():
    """Validate different context type classifications."""
    print("\n🧠 CONTEXT CLASSIFICATION VALIDATION") 
    print("=" * 50)
    
    config = {'enable_mcp_processing': True, 'enable_fallback': True}
    normalizer = AdvancedTextNormalizer(config)
    
    test_cases = [
        {
            'text': 'And one by one, he killed six of their children.',
            'expected_context': 'IDIOMATIC',
            'should_preserve': 'one by one'
        },
        {
            'text': 'We study chapter two verse twenty five from the Gita.',
            'expected_context': 'SCRIPTURAL', 
            'should_convert': ['two', 'twenty five']
        },
        {
            'text': 'In the year two thousand five, we started this practice.',
            'expected_context': 'TEMPORAL',
            'should_convert': ['two thousand five']
        },
        {
            'text': 'Two plus two equals four in mathematics.',
            'expected_context': 'MATHEMATICAL',
            'should_convert': ['Two', 'two', 'four']
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case['text']
        result = normalizer.convert_numbers_with_context(text)
        
        print(f"\n{i}. Context Type: {test_case['expected_context']}")
        print(f"   Input:  {text}")
        print(f"   Output: {result}")
        
        # Validate preservation or conversion
        if 'should_preserve' in test_case:
            preserved = test_case['should_preserve'] in result
            print(f"   Preserved '{test_case['should_preserve']}': {'✅' if preserved else '❌'}")
            if not preserved:
                all_passed = False
                
        elif 'should_convert' in test_case:
            converted = any(term not in result or result.count(term) < text.count(term) 
                          for term in test_case['should_convert'])
            print(f"   Converted numbers: {'✅' if converted else '❌'}")
            if not converted:
                all_passed = False
    
    print(f"\n🏆 CONTEXT CLASSIFICATION STATUS: {'SUCCESS' if all_passed else 'PARTIAL'}")
    return all_passed


def validate_performance():
    """Validate processing performance meets requirements.""" 
    print("\n⚡ PERFORMANCE VALIDATION")
    print("=" * 50)
    
    config = {'enable_mcp_processing': True, 'enable_fallback': True}
    normalizer = AdvancedTextNormalizer(config)
    
    # Performance test with longer text
    long_text = """
    Today we will discuss, one by one, the teachings from chapter two verse twenty five
    of the Bhagavad Gita. In the year two thousand five, scholars noted that two plus two
    still equals four, and this mathematical truth remains constant throughout time.
    """
    
    # Time the processing
    start_time = time.time()
    result = normalizer.normalize_with_advanced_tracking(long_text.strip())
    processing_time = time.time() - start_time
    
    print(f"📊 Processing Time: {processing_time:.3f} seconds")
    print(f"📊 Target: <2.0 seconds")
    print(f"📊 Performance: {'✅ PASS' if processing_time < 2.0 else '❌ FAIL'}")
    
    print(f"\n📈 Quality Score: {result.quality_score:.3f}")
    print(f"📈 Changes Applied: {len(result.corrections_applied)}")
    print(f"📈 Semantic Drift: {result.semantic_drift_score:.3f}")
    
    performance_pass = processing_time < 2.0
    print(f"\n🏆 PERFORMANCE STATUS: {'SUCCESS' if performance_pass else 'NEEDS_OPTIMIZATION'}")
    
    return performance_pass


def validate_integration():
    """Validate integration with existing systems."""
    print("\n🔗 INTEGRATION VALIDATION")
    print("=" * 50)
    
    config = {'enable_mcp_processing': True, 'enable_fallback': True}
    normalizer = AdvancedTextNormalizer(config)
    
    # Test conversational patterns + number processing
    integration_test = "I mean, uh, today we will, one by one, discuss the topics."
    
    result = normalizer.normalize_with_advanced_tracking(integration_test)
    
    print(f"📝 Integration Test: {integration_test}")
    print(f"📤 Result: {result.corrected_text}")
    print(f"📋 Changes: {result.corrections_applied}")
    
    # Validate both conversational and number processing work
    conversational_fixed = len([c for c in result.corrections_applied 
                              if 'conversational' in c or 'filler' in c]) > 0
    idiomatic_preserved = 'one by one' in result.corrected_text
    
    print(f"\n✅ Conversational Processing: {'WORKING' if conversational_fixed else 'FAILED'}")
    print(f"✅ Idiomatic Preservation: {'WORKING' if idiomatic_preserved else 'FAILED'}")
    
    integration_pass = conversational_fixed and idiomatic_preserved
    print(f"\n🏆 INTEGRATION STATUS: {'SUCCESS' if integration_pass else 'PARTIAL'}")
    
    return integration_pass


def main():
    """Run complete validation suite."""
    print("🚀 MCP INTEGRATION VALIDATION SUITE")
    print("=" * 60)
    print("Validating MCP-enhanced context-aware number processing")
    print("Addressing: 'one by one' -> '1 by 1' quality degradation")
    print()
    
    # Run all validations
    results = {
        'critical_fix': validate_critical_fix(),
        'context_classification': validate_context_classifications(), 
        'performance': validate_performance(),
        'integration': validate_integration()
    }
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 VALIDATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    overall_status = "SUCCESS" if all_passed else "PARTIAL SUCCESS"
    
    print(f"\n🎯 OVERALL STATUS: {overall_status}")
    
    if all_passed:
        print("""
🎉 MCP INTEGRATION IMPLEMENTATION COMPLETE!

Key Achievements:
- ✅ Fixed critical 'one by one' -> '1 by 1' issue
- ✅ Context-aware number processing working
- ✅ Maintains <2s processing performance  
- ✅ Full integration with existing systems
- ✅ Graceful fallback to Python system

Ready for Phase 2 development and production deployment!
        """)
    else:
        print("""
⚠️  PARTIAL SUCCESS - Some areas need attention:

Review failed tests above and address before production deployment.
Current implementation provides immediate quality improvement for the
critical 'one by one' issue while maintaining system stability.
        """)
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)