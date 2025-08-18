#!/usr/bin/env python3
"""
Debug Integration Test Failures
Find out exactly which validations are failing in the system integration test.
"""
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def debug_integration_test():
    """Debug the failing integration test to see which validations fail."""
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        # Create realistic test SRT content (same as in qa_independent_validation.py)
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
        
        # Process the file
        processor = SanskritPostProcessor()
        metrics = processor.process_srt_file(Path(temp_input), Path(temp_output))
        
        # Read processed content with Unicode handling
        with open(temp_output, 'r', encoding='utf-8') as f:
            processed_content = f.read()
        
        print("=== INPUT CONTENT ===")
        print(test_srt_content)
        print("\n=== PROCESSED CONTENT ===")
        try:
            print(processed_content)
        except UnicodeEncodeError:
            # Handle Unicode characters that can't be displayed in console
            safe_content = processed_content.encode('ascii', 'replace').decode('ascii')
            print(f"[Unicode content - showing ASCII approximation]")
            print(safe_content)
        print("\n=== METRICS ===")
        print(f"Total segments: {metrics.total_segments}")
        print(f"Segments modified: {metrics.segments_modified}")
        print(f"Processing time: {metrics.processing_time:.4f}s")
        print(f"Average confidence: {metrics.average_confidence:.3f}")
        
        # Check each validation individually
        print("\n=== VALIDATION CHECKS ===")
        validations = [
            ("Chapter 2 verse 25" in processed_content, "Scriptural conversion"),
            ("2005" in processed_content, "Year conversion"), 
            ("one by one" in processed_content, "Idiomatic preservation"),
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
            
            # Show details for failed checks
            if not check:
                if desc == "Scriptural conversion":
                    print(f"   Expected: 'Chapter 2 verse 25' in content")
                    print(f"   Found: Chapter occurrences: {processed_content.count('Chapter')}")
                elif desc == "Year conversion":
                    print(f"   Expected: '2005' in content")
                    print(f"   Found: '2005' count: {processed_content.count('2005')}")
                elif desc == "Idiomatic preservation":
                    print(f"   Expected: 'one by one' in content")
                    print(f"   Found: 'one by one' count: {processed_content.count('one by one')}")
                elif desc == "Mathematical conversion":
                    print(f"   Expected: '2 plus 2 equals 4' in content")
                    print(f"   Found: '2 plus 2' count: {processed_content.count('2 plus 2')}")
                elif desc == "NER processing":
                    print(f"   Expected: 'Krishna' or 'krishna' in content")
                    print(f"   Found: Krishna count: {processed_content.count('Krishna')}")
                    print(f"   Found: krishna count: {processed_content.count('krishna')}")
                elif desc == "Segment count":
                    print(f"   Expected: 4 segments")
                    print(f"   Found: {metrics.total_segments} segments")
                elif desc == "Modification tracking":
                    print(f"   Expected: >= 2 segments modified")
                    print(f"   Found: {metrics.segments_modified} segments modified")
        
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