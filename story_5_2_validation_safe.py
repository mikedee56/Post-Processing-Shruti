#!/usr/bin/env python3
"""
Story 5.2 Integration Remediation - Console-Safe Validation Test
================================================================

CRITICAL QA VALIDATION: Test all three fixes with safe console output.
"""

import sys
import logging
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Suppress verbose logging
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ['sanskrit_hindi_identifier', 'utils', 'post_processors', 'ner_module']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

def safe_print(text):
    """Print text safely by handling Unicode characters."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters with safe representations
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

def test_all_fixes_integrated():
    """Test all three fixes together in the integrated pipeline."""
    safe_print("=== Story 5.2 Integration Remediation - Final Validation ===")
    safe_print("")
    
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    
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
        
        safe_print("Processing comprehensive test SRT...")
        
        # Process the file
        metrics = processor.process_srt_file(Path(temp_input), Path(temp_output))
        
        safe_print(f"Processing completed:")
        safe_print(f"  Total segments: {metrics.total_segments}")
        safe_print(f"  Segments modified: {metrics.segments_modified}")
        safe_print(f"  Processing time: {metrics.processing_time:.4f}s")
        safe_print(f"  Average confidence: {metrics.average_confidence:.3f}")
        safe_print("")
        
        # Read and validate processed output
        if os.path.exists(temp_output):
            with open(temp_output, 'r', encoding='utf-8') as f:
                processed_content = f.read()
            
            # Critical QA validations (handle Unicode safely)
            validations = []
            
            # Fix 1 - Scriptural conversion
            scriptural_pass = "Chapter 2 verse 25" in processed_content
            validations.append((scriptural_pass, "Scriptural conversion: chapter two verse twenty five -> Chapter 2 verse 25"))
            
            # Fix 2 & 3 - Sanskrit capitalization (without corruption)
            krishna_pass = "Krishna" in processed_content
            validations.append((krishna_pass, "Sanskrit capitalization: krishna -> Krishna (no corruption)"))
            
            dharma_pass = "Dharma" in processed_content  
            validations.append((dharma_pass, "Sanskrit capitalization: dharma -> Dharma"))
            
            vishnu_pass = "Vishnu" in processed_content
            validations.append((vishnu_pass, "Sanskrit capitalization: vishnu -> Vishnu"))
            
            shiva_pass = "Shiva" in processed_content
            validations.append((shiva_pass, "Sanskrit capitalization: shiva -> Shiva"))
            
            # Fix 2 - No over-capitalization of common words
            verse_ok = "verse" in processed_content.lower() and "Verse" not in processed_content
            validations.append((verse_ok, "No over-capitalization of common word 'verse'"))
            
            # Fix 1 - Idiomatic preservation
            idiomatic_pass = "one by one" in processed_content.lower()
            validations.append((idiomatic_pass, "Idiomatic preservation: one by one maintained"))
            
            # Additional quality checks
            no_corruption = "K???a" not in processed_content and "???" not in processed_content
            validations.append((no_corruption, "No Unicode corruption detected"))
            
            no_numeric_idiom = "1 by 1" not in processed_content
            validations.append((no_numeric_idiom, "No incorrect numeric conversion of idioms"))
            
            safe_print("CRITICAL QA VALIDATION RESULTS:")
            all_validations_pass = True
            for passed, description in validations:
                status = "PASS" if passed else "FAIL"
                if not passed:
                    all_validations_pass = False
                safe_print(f"  {status}: {description}")
            
            safe_print("")
            
            # Show safe representation of content
            safe_print("Processed content (first 200 characters):")
            safe_content = processed_content[:200].encode('ascii', errors='replace').decode('ascii')
            safe_print(f"  {repr(safe_content)}")
            safe_print("")
            
            return all_validations_pass
            
        else:
            safe_print("ERROR: Processed file not created")
            return False
    
    except Exception as e:
        safe_print(f"ERROR: Integration test failed - {e}")
        return False
    
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
    """Run the safe validation test."""
    success = test_all_fixes_integrated()
    
    safe_print("=" * 60)
    safe_print("FINAL STORY 5.2 VALIDATION RESULTS")
    safe_print("=" * 60)
    safe_print("")
    
    if success:
        safe_print("SUCCESS: Story 5.2 Integration Remediation COMPLETE")
        safe_print("All critical QA failures have been resolved!")
        safe_print("System is ready for production approval.")
        safe_print("")
        safe_print("KEY ACHIEVEMENTS:")
        safe_print("• Scriptural conversions working correctly")
        safe_print("• Sanskrit capitalization with no Unicode corruption")
        safe_print("• Idiomatic expressions properly preserved")
        safe_print("• NER over-capitalization eliminated")
        safe_print("• All components integrated successfully")
        
        return True
    else:
        safe_print("FAILURE: Some critical issues remain unresolved")
        safe_print("Manual investigation required for failed components.")
        safe_print("System NOT ready for production deployment.")
        
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    exit(exit_code)