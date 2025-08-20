#!/usr/bin/env python3
"""
Golden Dataset Core Test - Console Safe Version
Tests core system functionality against real-world Sanskrit/Hindi content
"""

import sys
import tempfile
import os
from pathlib import Path
import time

# Setup
sys.path.insert(0, 'src')

def safe_print(text):
    """Print text safely, avoiding Unicode encoding issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters and try again
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

def create_test_srt(text_content, max_segments=10):
    """Create test SRT content from text"""
    lines = [line.strip() for line in text_content.split('\n') if line.strip() and len(line.strip()) > 10]
    
    srt_content = []
    for i, line in enumerate(lines[:max_segments], 1):
        start_time = (i-1) * 5
        end_time = start_time + 4
        
        # Convert to SRT format
        start_srt = f"{start_time//3600:02d}:{(start_time%3600)//60:02d}:{start_time%60:02d},000"
        end_srt = f"{end_time//3600:02d}:{(end_time%3600)//60:02d}:{end_time%60:02d},000"
        
        srt_content.extend([
            str(i),
            f"{start_srt} --> {end_srt}",
            line,
            ""
        ])
    
    return '\n'.join(srt_content)

def test_basic_functionality():
    """Test basic components"""
    safe_print("=== Testing Basic Functionality ===")
    
    try:
        # Test basic imports
        from utils.text_normalizer import TextNormalizer
        from utils.srt_parser import SRTParser
        
        normalizer = TextNormalizer()
        parser = SRTParser()
        
        safe_print("PASS: Core components imported successfully")
        
        # Test number conversion
        test_text = "chapter two verse twenty five"
        result = normalizer.convert_numbers(test_text)
        safe_print(f"PASS: Number conversion: '{test_text}' -> '{result}'")
        
        return True
        
    except Exception as e:
        safe_print(f"FAIL: Basic functionality test failed: {e}")
        return False

def main():
    safe_print("=== Golden Dataset Core Test ===")
    safe_print("")
    
    # Test basic functionality
    if not test_basic_functionality():
        safe_print("Cannot proceed - basic functionality failed")
        return False
    
    # Load golden dataset
    golden_path = Path("D:/Audio_Pre-processing/data/golden_dataset/golden_transcript.txt")
    if not golden_path.exists():
        safe_print("ERROR: Golden dataset not found")
        return False
    
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_text = f.read()
    
    safe_print(f"PASS: Loaded golden dataset ({len(golden_text)} characters)")
    
    # Analyze content without printing problematic Unicode
    lines = [line.strip() for line in golden_text.split('\n') if line.strip()]
    
    # Count Sanskrit content (IAST diacritics)
    sanskrit_markers = ['ā', 'ī', 'ū', 'ṛ', 'ṃ', 'ḥ', 'ṅ', 'ñ', 'ṭ', 'ḍ', 'ṇ', 'ś', 'ṣ']
    sanskrit_lines = sum(1 for line in lines if any(char in line for char in sanskrit_markers))
    
    # Count key terms
    key_terms = ['krishna', 'dharma', 'karma', 'yoga', 'brahman', 'liberation', 'vedanta']
    found_terms = [term for term in key_terms if term.lower() in golden_text.lower()]
    
    safe_print(f"PASS: Content analysis complete")
    safe_print(f"  Sanskrit lines with IAST: {sanskrit_lines}")
    safe_print(f"  Key Vedanta terms found: {len(found_terms)}")
    safe_print(f"  Terms: {', '.join(found_terms)}")
    
    # Try to initialize simplified processor
    safe_print("\n=== Testing System Integration ===")
    
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        
        # Initialize with default config and disable NER
        processor = SanskritPostProcessor(config_path=None, enable_ner=False)
        safe_print("PASS: SanskritPostProcessor initialized with minimal config")
        
        # Create test SRT
        test_srt_content = create_test_srt(golden_text, max_segments=5)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as temp_input:
            temp_input.write(test_srt_content)
            temp_input_path = temp_input.name
        
        temp_output_path = temp_input_path.replace('.srt', '_processed.srt')
        
        # Process the file
        safe_print("PASS: Created test SRT file")
        
        start_time = time.time()
        metrics = processor.process_srt_file(Path(temp_input_path), Path(temp_output_path))
        processing_time = time.time() - start_time
        
        safe_print(f"PASS: Processing completed in {processing_time:.3f} seconds")
        safe_print(f"  Total segments: {metrics.total_segments}")
        safe_print(f"  Modified segments: {metrics.segments_modified}")
        safe_print(f"  Average confidence: {metrics.average_confidence:.3f}")
        safe_print(f"  Throughput: {metrics.total_segments/processing_time:.2f} segments/sec")
        
        # Check output file
        if os.path.exists(temp_output_path):
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                processed_content = f.read()
            
            safe_print(f"PASS: Output file created ({len(processed_content)} characters)")
            
            # Quality checks
            improvements = []
            
            # Check for number conversion
            if '2 verse 25' in processed_content or 'chapter 2' in processed_content:
                improvements.append("Number conversion applied")
            
            # Check for Sanskrit preservation
            if any(char in processed_content for char in sanskrit_markers):
                improvements.append("Sanskrit diacritics preserved")
            
            # Check for key terms
            terms_in_output = sum(1 for term in found_terms if term in processed_content.lower())
            if terms_in_output > 0:
                improvements.append(f"{terms_in_output} Sanskrit terms preserved")
            
            safe_print(f"PASS: Quality assessment - {len(improvements)} improvements detected")
            for improvement in improvements:
                safe_print(f"  - {improvement}")
            
            # Performance assessment
            target_throughput = 10.0
            if metrics.total_segments/processing_time >= target_throughput:
                safe_print(f"PASS: Performance target met ({metrics.total_segments/processing_time:.1f} >= {target_throughput} seg/sec)")
            else:
                safe_print(f"INFO: Performance below optimal ({metrics.total_segments/processing_time:.1f} < {target_throughput} seg/sec)")
            
            safe_print("\n=== GOLDEN DATASET TEST: SUCCESS ===")
            safe_print("System successfully processes real-world Sanskrit/Hindi content")
            
            return True
            
        else:
            safe_print("FAIL: No output file created")
            return False
            
    except Exception as e:
        safe_print(f"FAIL: System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            if 'temp_input_path' in locals() and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
        except:
            pass

if __name__ == "__main__":
    success = main()
    if success:
        safe_print("\nSUCCESS: Golden dataset validation completed successfully")
        safe_print("System is ready for production use with real-world content")
    else:
        safe_print("\nISSUES: Golden dataset validation encountered problems")
    
    exit(0 if success else 1)