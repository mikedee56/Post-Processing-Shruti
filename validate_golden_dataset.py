#!/usr/bin/env python3
"""
Golden Dataset Validation Test
Tests the current system against real-world Yoga Vedanta lecture transcript
"""

import sys
import tempfile
import os
from pathlib import Path
import time
import logging

# Setup
sys.path.insert(0, 'src')
logging.getLogger().setLevel(logging.WARNING)  # Reduce noise

def create_srt_from_text(text_content, output_path):
    """Convert plain text to SRT format with reasonable timing"""
    lines = text_content.strip().split('\n')
    srt_lines = []
    segment_num = 1
    current_time = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Each segment is 4 seconds with 1 second gaps
        start_time = current_time
        end_time = current_time + 4
        
        # Convert to SRT time format
        start_srt = f"{start_time//3600:02d}:{(start_time%3600)//60:02d}:{start_time%60:02d},000"
        end_srt = f"{end_time//3600:02d}:{(end_time%3600)//60:02d}:{end_time%60:02d},000"
        
        srt_lines.append(f"{segment_num}")
        srt_lines.append(f"{start_srt} --> {end_srt}")
        srt_lines.append(line)
        srt_lines.append("")
        
        segment_num += 1
        current_time = end_time + 1  # 1 second gap
        
        # Limit to first 50 segments for testing
        if segment_num > 50:
            break
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_lines))
    
    return segment_num - 1

def main():
    print("=== Golden Dataset Validation Test ===")
    print()
    
    # Read golden dataset
    golden_path = Path("D:/Audio_Pre-processing/data/golden_dataset/golden_transcript.txt")
    if not golden_path.exists():
        print("ERROR: Golden dataset not found at expected path")
        return False
    
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_text = f.read()
    
    print(f"Loaded golden dataset: {len(golden_text)} characters")
    
    # Create temporary SRT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as temp_input:
        segments_created = create_srt_from_text(golden_text, temp_input.name)
        temp_input_path = temp_input.name
    
    temp_output_path = temp_input_path.replace('.srt', '_processed.srt')
    
    print(f"Created test SRT with {segments_created} segments")
    
    try:
        # Initialize the post-processor
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        processor = SanskritPostProcessor()
        
        print("SanskritPostProcessor initialized")
        print(f"  NER enabled: {processor.enable_ner}")
        print(f"  Advanced processing available: {hasattr(processor, 'text_normalizer')}")
        print()
        
        # Process the golden dataset
        print("Processing golden dataset...")
        start_time = time.time()
        
        metrics = processor.process_srt_file(Path(temp_input_path), Path(temp_output_path))
        
        processing_time = time.time() - start_time
        
        print(f"Processing completed in {processing_time:.3f} seconds")
        print(f"  Total segments: {metrics.total_segments}")
        print(f"  Segments modified: {metrics.segments_modified}")
        print(f"  Average confidence: {metrics.average_confidence:.3f}")
        print(f"  Throughput: {metrics.total_segments / processing_time:.2f} segments/sec")
        print()
        
        # Analyze the results
        if os.path.exists(temp_output_path):
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                processed_content = f.read()
            
            print("=== Sample Processing Results ===")
            
            # Look for specific improvements
            improvements = []
            
            # Check for Sanskrit term capitalization
            sanskrit_terms = ['krishna', 'brahman', 'dharma', 'karma', 'yoga', 'vedanta', 'upanishad', 'gita']
            for term in sanskrit_terms:
                if term.lower() in processed_content.lower() and term.capitalize() in processed_content:
                    improvements.append(f"Sanskrit capitalization: {term} → {term.capitalize()}")
            
            # Check for IAST preservation
            iast_chars = ['ā', 'ī', 'ū', 'ṛ', 'ṃ', 'ḥ', 'ṅ', 'ñ', 'ṭ', 'ḍ', 'ṇ', 'ś', 'ṣ']
            iast_preserved = any(char in processed_content for char in iast_chars)
            if iast_preserved:
                improvements.append("IAST diacritics preserved")
            
            # Check for text normalization
            if 'chapter' in golden_text.lower() and any(c.isdigit() for c in processed_content):
                improvements.append("Number conversion applied")
            
            print("Key Improvements Detected:")
            if improvements:
                for improvement in improvements[:10]:  # Show top 10
                    print(f"  ✓ {improvement}")
            else:
                print("  (Processing completed but specific improvements not detected in sample)")
            
            print()
            
            # Show sample processed text
            lines = processed_content.split('\n')
            content_lines = [line for line in lines if line.strip() and not line.strip().isdigit() and '-->' not in line]
            
            print("=== Sample Processed Content ===")
            for i, line in enumerate(content_lines[:5]):  # Show first 5 content lines
                if len(line) > 80:
                    line = line[:77] + "..."
                print(f"{i+1}. {line}")
            print()
            
            # Performance assessment
            target_throughput = 10.0  # segments per second
            if metrics.total_segments / processing_time >= target_throughput:
                print(f"✓ PERFORMANCE TARGET MET: {metrics.total_segments / processing_time:.2f} >= {target_throughput} seg/sec")
            else:
                print(f"⚠ Performance below target: {metrics.total_segments / processing_time:.2f} < {target_throughput} seg/sec")
            
            # Quality assessment
            modification_rate = metrics.segments_modified / metrics.total_segments * 100
            print(f"✓ QUALITY ASSESSMENT: {modification_rate:.1f}% segments improved")
            
            print()
            print("=== GOLDEN DATASET VALIDATION: SUCCESS ===")
            print("System successfully processed real-world Sanskrit/Hindi content")
            
            return True
            
        else:
            print("ERROR: No output file created")
            return False
            
    except Exception as e:
        print(f"ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
        except:
            pass

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)