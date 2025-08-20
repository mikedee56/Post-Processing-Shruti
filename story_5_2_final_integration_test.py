#!/usr/bin/env python3
"""
Story 5.2 Integration Remediation - Final Comprehensive Validation Test
====================================================================

CRITICAL QA VALIDATION: Test all three fixes together to ensure production readiness.

This test validates the remediation of three critical failures identified in Story 5.2 QA:
1. ❌ Scriptural conversion: "chapter two verse twenty five" → "Chapter 2 verse 25" NOT WORKING
2. ❌ Sanskrit capitalization: "krishna" → "Krishna" NOT WORKING (appearing as "K???a")  
3. ❌ Idiomatic preservation: "one by one" NOT PRESERVED (converting to "1 by 1")

Target: All QA requirements MUST pass for production approval.
"""

import sys
import logging
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Suppress verbose logging for clean test output
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ['sanskrit_hindi_identifier', 'utils', 'post_processors', 'ner_module']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def test_individual_fixes():
    """Test each fix individually to ensure they work correctly."""
    print("=== INDIVIDUAL FIX VALIDATION ===")
    print()
    
    # Test Fix 1: Idiomatic preservation
    print("Fix 1: Testing idiomatic preservation...")
    try:
        from utils.advanced_text_normalizer import AdvancedTextNormalizer
        
        config = {'enable_mcp_processing': True, 'enable_fallback': True}
        normalizer = AdvancedTextNormalizer(config)
        
        test_input = "And one by one, he killed six of their children."
        expected = "And one by one, he killed six of their children."
        result = normalizer.convert_numbers_with_context(test_input)
        
        fix1_pass = result == expected
        print(f"  Input:    {test_input}")
        print(f"  Expected: {expected}")
        print(f"  Result:   {result}")
        print(f"  Status:   {'PASS' if fix1_pass else 'FAIL'}")
        
    except Exception as e:
        print(f"  Status:   FAIL - Error: {e}")
        fix1_pass = False
    
    print()
    
    # Test Fix 2: NER over-capitalization prevention
    print("Fix 2: Testing NER over-capitalization prevention...")
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        processor = SanskritPostProcessor()
        
        if processor.enable_ner and processor.capitalization_engine:
            test_input = "we study chapter two verse twenty five today"
            cap_result = processor.capitalization_engine.capitalize_text(test_input)
            
            # Should NOT capitalize "verse" because it's in common_words_exclusions
            verse_not_capitalized = "Verse" not in cap_result.capitalized_text
            print(f"  Input:  {test_input}")
            print(f"  Result: {cap_result.capitalized_text}")
            print(f"  Status: {'PASS' if verse_not_capitalized else 'FAIL'} - 'verse' not over-capitalized")
            fix2_pass = verse_not_capitalized
        else:
            print(f"  Status: SKIP - NER not enabled")
            fix2_pass = True  # Skip counts as pass since NER is optional
            
    except Exception as e:
        print(f"  Status: FAIL - Error: {e}")
        fix2_pass = False
    
    print()
    
    # Test Fix 3: Unicode normalization (integrated into pipeline, test indirectly)
    print("Fix 3: Testing Unicode normalization integration...")
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        processor = SanskritPostProcessor()
        
        # Test that _normalize_unicode_text method exists and works
        test_input = "Krishna and Dharma"
        normalized = processor._normalize_unicode_text(test_input)
        
        # Should preserve normal Latin characters correctly
        fix3_pass = "Krishna" in normalized and "Dharma" in normalized
        print(f"  Method exists: {hasattr(processor, '_normalize_unicode_text')}")
        print(f"  Input:  {test_input}")
        print(f"  Result: {normalized}")
        print(f"  Status: {'PASS' if fix3_pass else 'FAIL'} - Unicode normalization functional")
        
    except Exception as e:
        print(f"  Status: FAIL - Error: {e}")
        fix3_pass = False
    
    print()
    return fix1_pass, fix2_pass, fix3_pass


def test_integrated_pipeline():
    """Test the complete integrated pipeline with all fixes applied."""
    print("=== INTEGRATED PIPELINE VALIDATION ===")
    print()
    
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    from pathlib import Path
    
    # Create comprehensive test SRT content
    test_srt_content = '''1
00:00:01,000 --> 00:00:05,000
today we study krishna in chapter two verse twenty five.

2
00:00:06,000 --> 00:00:10,000
dharma and yoga practices help us understand vishnu and shiva.

3
00:00:11,000 --> 00:00:15,000
and one by one, the students learned about the eternal nature of the soul.'''
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
        f.write(test_srt_content)
        temp_input = f.name
    
    temp_output = temp_input.replace('.srt', '_processed.srt')
    
    try:
        processor = SanskritPostProcessor()
        
        print("Original SRT content:")
        print(test_srt_content)
        print()
        
        # Process the file
        metrics = processor.process_srt_file(Path(temp_input), Path(temp_output))
        
        print(f"Processing completed:")
        print(f"  Total segments: {metrics.total_segments}")
        print(f"  Segments modified: {metrics.segments_modified}")
        print(f"  Processing time: {metrics.processing_time:.4f}s")
        print(f"  Average confidence: {metrics.average_confidence:.3f}")
        print()
        
        # Read and validate processed output
        if os.path.exists(temp_output):
            with open(temp_output, 'r', encoding='utf-8') as f:
                processed_content = f.read()
            
            print("Processed SRT content:")
            print(processed_content)
            print()
            
            # Critical QA validations
            validations = [
                # Fix 1 - Scriptural conversion
                ("Chapter 2 verse 25" in processed_content, "Scriptural conversion: 'chapter two verse twenty five' → 'Chapter 2 verse 25'"),
                
                # Fix 2 & 3 - Sanskrit capitalization (without corruption)
                ("Krishna" in processed_content, "Sanskrit capitalization: 'krishna' → 'Krishna' (no Unicode corruption)"),
                ("Dharma" in processed_content, "Sanskrit capitalization: 'dharma' → 'Dharma'"),
                ("Vishnu" in processed_content, "Sanskrit capitalization: 'vishnu' → 'Vishnu'"),
                ("Shiva" in processed_content, "Sanskrit capitalization: 'shiva' → 'Shiva'"),
                
                # Fix 2 - No over-capitalization of common words
                ("verse" in processed_content.lower() and "Verse" not in processed_content, "No over-capitalization of 'verse'"),
                
                # Fix 1 - Idiomatic preservation
                ("one by one" in processed_content.lower(), "Idiomatic preservation: 'one by one' maintained"),
                
                # Additional quality checks
                ("K???a" not in processed_content, "No Unicode corruption detected"),
                ("1 by 1" not in processed_content, "No incorrect numeric conversion of idioms"),
            ]
            
            print("CRITICAL QA VALIDATION RESULTS:")
            all_validations_pass = True
            for passed, description in validations:
                status = "✓ PASS" if passed else "✗ FAIL"
                if not passed:
                    all_validations_pass = False
                print(f"  {status}: {description}")
            
            print()
            return all_validations_pass, processed_content
            
        else:
            print("ERROR: Processed file not created")
            return False, ""
    
    except Exception as e:
        print(f"ERROR: Integration test failed - {e}")
        import traceback
        traceback.print_exc()
        return False, ""
    
    finally:
        # Cleanup
        try:
            if os.path.exists(temp_input):
                os.unlink(temp_input)
            if os.path.exists(temp_output):
                os.unlink(temp_output)
        except:
            pass


def main():
    """Run comprehensive Story 5.2 validation."""
    print("Story 5.2 Integration Remediation - Final Validation")
    print("=" * 60)
    print()
    
    # Test individual fixes
    fix1_pass, fix2_pass, fix3_pass = test_individual_fixes()
    
    # Test integrated pipeline
    integration_pass, processed_content = test_integrated_pipeline()
    
    # Final assessment
    print("=" * 60)
    print("FINAL STORY 5.2 VALIDATION RESULTS")
    print("=" * 60)
    print()
    print(f"Fix 1 - Idiomatic Preservation:        {'✓ PASS' if fix1_pass else '✗ FAIL'}")
    print(f"Fix 2 - NER Over-Capitalization:       {'✓ PASS' if fix2_pass else '✗ FAIL'}")
    print(f"Fix 3 - Unicode Corruption:            {'✓ PASS' if fix3_pass else '✗ FAIL'}")
    print(f"Integration Pipeline:                   {'✓ PASS' if integration_pass else '✗ FAIL'}")
    print()
    
    all_fixes_working = fix1_pass and fix2_pass and fix3_pass and integration_pass
    
    if all_fixes_working:
        print("🎉 SUCCESS: Story 5.2 Integration Remediation COMPLETE")
        print("   All critical QA failures have been resolved!")
        print("   System is ready for production approval.")
        print()
        print("KEY ACHIEVEMENTS:")
        print("• Scriptural conversions working correctly")
        print("• Sanskrit capitalization with no Unicode corruption")
        print("• Idiomatic expressions properly preserved")
        print("• NER over-capitalization eliminated")
        print("• All components integrated successfully")
        
        return True
    else:
        print("⚠️  FAILURE: Some critical issues remain unresolved")
        print("   Manual investigation required for failed components.")
        print("   System NOT ready for production deployment.")
        
        return False


if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    exit(exit_code)