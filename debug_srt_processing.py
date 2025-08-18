#!/usr/bin/env python3
"""
Debug SRT Processing Pipeline
Check what's happening at each stage of the processing
"""
import sys
import tempfile
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, 'src')

# Disable all logging
logging.getLogger().setLevel(logging.CRITICAL)

def debug_srt_processing():
    """Debug the SRT processing pipeline step by step."""
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        from utils.srt_parser import SRTParser
        
        # Focus on the problematic segment
        test_srt_content = '''3
00:00:13,000 --> 00:00:18,000
One by one, the students learned that two plus two equals four in mathematics.
'''
        
        # Parse the SRT content
        parser = SRTParser()
        segments = parser.parse_string(test_srt_content)
        segment = segments[0]
        
        print("=== SRT PROCESSING PIPELINE DEBUG ===")
        print(f"Original segment text: \"{segment.text}\"")
        
        # Initialize processor
        processor = SanskritPostProcessor(config_path=None, enable_ner=False)
        
        # Test the text normalizer directly
        print("\n=== DIRECT TEXT NORMALIZER TEST ===")
        if hasattr(processor.text_normalizer, 'convert_numbers_with_context'):
            direct_result = processor.text_normalizer.convert_numbers_with_context(segment.text)
            print(f"Direct normalizer result: \"{direct_result}\"")
        else:
            print("No convert_numbers_with_context method available")
        
        # Test the full segment processing
        print("\n=== FULL SEGMENT PROCESSING ===")
        metrics = processor.metrics_collector.create_file_metrics('debug')
        processed_segment = processor._process_srt_segment(segment, metrics)
        print(f"Processed segment text: \"{processed_segment.text}\"")
        
        # Compare results
        print("\n=== COMPARISON ===")
        print(f"Original:  \"{segment.text}\"")
        if hasattr(processor.text_normalizer, 'convert_numbers_with_context'):
            print(f"Direct:    \"{direct_result}\"")
        print(f"Pipeline:  \"{processed_segment.text}\"")
        
        # Check what changes
        if hasattr(processor.text_normalizer, 'convert_numbers_with_context'):
            direct_preserves_idiom = 'One by one' in direct_result
            direct_converts_math = '2 plus 2 equals 4' in direct_result
            print(f"\nDirect normalizer: preserves idiom={direct_preserves_idiom}, converts math={direct_converts_math}")
        
        pipeline_preserves_idiom = 'One by one' in processed_segment.text
        pipeline_converts_math = '2 plus 2 equals 4' in processed_segment.text
        print(f"Pipeline result: preserves idiom={pipeline_preserves_idiom}, converts math={pipeline_converts_math}")
        
        return True
        
    except Exception as e:
        print(f"DEBUG FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_srt_processing()