#!/usr/bin/env python3
"""
Final Integration Debug - Focus on Individual Segments
Debug each segment transformation to identify specific issues.
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

def debug_individual_segments():
    """Debug each segment individually to understand transformations."""
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        from utils.srt_parser import SRTParser
        
        # Create test segments one by one
        test_segments = [
            ("1", "Chapter two verse twenty five", "SCRIPTURAL"),
            ("2", "In the year two thousand five", "TEMPORAL"), 
            ("3", "One by one, the students learned that two plus two equals four", "MIXED"),
            ("4", "The ancient teachers like krishna and patanjali", "NER")
        ]
        
        processor = SanskritPostProcessor(config_path=None, enable_ner=False)
        
        print("=== INDIVIDUAL SEGMENT DEBUG ===")
        for seg_id, text, context in test_segments:
            print(f"\n{seg_id}. {context}: \"{text}\"")
            
            # Create individual SRT content with proper formatting
            start_time = f"00:00:0{seg_id},000"
            end_time = f"00:00:{'0' if int(seg_id)+4 < 10 else ''}{int(seg_id)+4},000"
            srt_content = f"""{seg_id}
{start_time} --> {end_time}
{text}

"""
            
            # Parse and process
            parser = SRTParser()
            segments = parser.parse_string(srt_content)
            segment = segments[0]
            
            # Process the segment
            metrics = processor.metrics_collector.create_file_metrics('debug')
            processed_segment = processor._process_srt_segment(segment, metrics)
            
            original = segment.text
            processed = processed_segment.text
            changed = original != processed
            
            # Use ASCII-safe output to avoid Unicode issues
            try:
                print(f"   Original:  {repr(original)}")
                print(f"   Processed: {repr(processed)}")
                print(f"   Changed:   {changed}")
                
                # Specific validation checks for each context
                if context == "SCRIPTURAL":
                    has_chapter_2 = "Chapter 2" in processed
                    has_verse_25 = "verse 25" in processed
                    print(f"   Chapter 2: {has_chapter_2}, verse 25: {has_verse_25}")
                elif context == "TEMPORAL":
                    has_2005 = "2005" in processed
                    print(f"   Year 2005: {has_2005}")
                elif context == "MIXED":
                    preserves_idiom = "one by one" in processed.lower()
                    converts_math = "2 plus 2" in processed
                    print(f"   Preserves idiom: {preserves_idiom}, Converts math: {converts_math}")
                elif context == "NER":
                    has_krishna = "krishna" in processed.lower() or "Krishna" in processed
                    print(f"   Has krishna: {has_krishna}")
                    
            except UnicodeEncodeError:
                # Fallback for Unicode issues
                print(f"   Original length:  {len(original)}")
                print(f"   Processed length: {len(processed)}")
                print(f"   Changed:   {changed}")
        
        return True
        
    except Exception as e:
        print(f"DEBUG FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_individual_segments()