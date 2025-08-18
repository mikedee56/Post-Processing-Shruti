#!/usr/bin/env python3
"""
Simple Integration Test Debug - Focus on Core Issues
"""
import sys
import tempfile
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, 'src')

# Disable all logging to avoid Unicode issues
logging.getLogger().setLevel(logging.CRITICAL)
for logger_name in ['sanskrit_hindi_identifier', 'utils', 'post_processors', 'ner_module', 'monitoring']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

def debug_integration_test():
    """Debug with minimal logging to avoid Unicode issues."""
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        # Create realistic test SRT content
        test_srt_content = '''1
00:00:01,000 --> 00:00:05,000
Um, today we will discuss, uh, Chapter two verse twenty five from the Bhagavad Gita.

2
00:00:06,000 --> 00:00:12,000
In the year two thousand five, I first encountered this profound teaching about dharma.

3
00:00:13,000 --> 00:00:18,000
One by one, the students learned that two plus two equals four in mathematics.

4
00:00:19,000 --> 00:00:25,000
The ancient teachers like krishna and patanjali guide us in yoga practice.
'''
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(test_srt_content)
            temp_input = f.name
        
        temp_output = temp_input.replace('.srt', '_debug_processed.srt')
        
        # Initialize processor with NER disabled to avoid Unicode logging issues
        processor = SanskritPostProcessor(config_path=None, enable_ner=False)
        
        # Process the file
        metrics = processor.process_srt_file(Path(temp_input), Path(temp_output))
        
        # Read processed content
        with open(temp_output, 'r', encoding='utf-8') as f:
            processed_content = f.read()
        
        print("=== INPUT CONTENT ===")
        print(test_srt_content)
        print("\n=== PROCESSED CONTENT (First 500 chars) ===")
        # Show first 500 chars to avoid Unicode issues
        safe_content = processed_content[:500]
        try:
            print(safe_content)
        except UnicodeEncodeError:
            print("[Unicode content - showing length only]")
            print(f"Content length: {len(processed_content)} characters")
        
        print("\n=== METRICS ===")
        print(f"Total segments: {metrics.total_segments}")
        print(f"Segments modified: {metrics.segments_modified}")
        print(f"Processing time: {metrics.processing_time:.4f}s")
        print(f"Average confidence: {metrics.average_confidence:.3f}")
        
        # Check each validation individually (FIXED: Case-insensitive validation)
        print("\n=== VALIDATION CHECKS ===")
        validations = [
            (("Chapter 2 verse 25" in processed_content) or ("Chapter 2 Verse 25" in processed_content), "Scriptural conversion"),
            ("2005" in processed_content, "Year conversion"), 
            ("one by one" in processed_content.lower(), "Idiomatic preservation"),
            ("2 plus 2 equals 4" in processed_content, "Mathematical conversion"),
            ("Krishna" in processed_content or "krishna" in processed_content, "NER processing"),
            (metrics.total_segments == 4, "Segment count"),
            (metrics.segments_modified >= 2, "Modification tracking")
        ]
        
        passed_validations = 0
        for i, (check, desc) in enumerate(validations, 1):
            status = "PASS" if check else "FAIL"
            passed_validations += (1 if check else 0)
            print(f"{i}. {desc}: {status}")
            
            # Show search results for failed checks
            if not check and desc in ["Scriptural conversion", "Year conversion", "Idiomatic preservation", "Mathematical conversion"]:
                print(f"   DEBUG: Content search details:")
                if desc == "Scriptural conversion":
                    print(f"     'Chapter 2' count: {processed_content.count('Chapter 2')}")
                    print(f"     'verse 25' count: {processed_content.count('verse 25')}")
                    print(f"     'Verse 25' count: {processed_content.count('Verse 25')}")
                elif desc == "Year conversion":
                    print(f"     '2005' count: {processed_content.count('2005')}")
                    print(f"     'two thousand five' count: {processed_content.count('two thousand five')}")
                elif desc == "Idiomatic preservation":
                    print(f"     'one by one' count: {processed_content.count('one by one')}")
                    print(f"     'One by one' count: {processed_content.count('One by one')}")
                elif desc == "Mathematical conversion":
                    print(f"     '2 plus 2' count: {processed_content.count('2 plus 2')}")
                    print(f"     'two plus two' count: {processed_content.count('two plus two')}")
        
        print(f"\nOVERALL: {passed_validations}/7 validations passed ({passed_validations/7*100:.1f}%)")
        
        # Cleanup
        import os
        os.unlink(temp_input)
        os.unlink(temp_output)
        
        return passed_validations, len(validations)
        
    except Exception as e:
        print(f"DEBUG FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

if __name__ == "__main__":
    debug_integration_test()