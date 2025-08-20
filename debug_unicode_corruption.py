#!/usr/bin/env python3
"""
Debug Unicode Corruption - Story 5.2 Fix 3
===========================================

CRITICAL ISSUE: Sanskrit terms like "Krishna" are appearing as "K???a" in output.
This suggests Unicode encoding/decoding issues during processing.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Suppress verbose logging
logging.getLogger().setLevel(logging.ERROR)

def debug_unicode_at_each_step():
    """Debug Unicode handling at each processing step."""
    print("=== UNICODE CORRUPTION DEBUGGING ===")
    print()
    
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    from utils.srt_parser import SRTSegment
    import copy
    
    processor = SanskritPostProcessor()
    
    # Create test segment
    test_text = "today we study krishna and dharma"
    segment = SRTSegment(
        index=1,
        start_time="00:00:01,000",
        end_time="00:00:05,000", 
        text=test_text,
        raw_text=test_text
    )
    
    print(f"ORIGINAL: {repr(test_text)}")
    print()
    
    # Track text at each step
    processed_segment = copy.deepcopy(segment)
    
    # Step 1: Enhanced Text Normalization
    print("=== STEP 1: Enhanced Text Normalization ===")
    if hasattr(processor.text_normalizer, 'normalize_with_advanced_tracking'):
        advanced_result = processor.text_normalizer.normalize_with_advanced_tracking(processed_segment.text)
        processed_segment.text = advanced_result.corrected_text
        print(f"After normalization: {repr(processed_segment.text)}")
        # Check for corruption
        if "krishna" not in processed_segment.text.lower():
            print("⚠️  WARNING: 'krishna' modified or corrupted in normalization step")
    print()
    
    # Step 2: Enhanced contextual number processing
    print("=== STEP 2: Contextual Number Processing ===")
    number_result = processor.number_processor.process_numbers(processed_segment.text, context="spiritual")
    processed_segment.text = number_result.processed_text
    print(f"After number processing: {repr(processed_segment.text)}")
    # Check for corruption
    if "krishna" not in processed_segment.text.lower() and "k" in processed_segment.text.lower():
        print("⚠️  WARNING: Possible Unicode corruption in number processing")
    print()
    
    # Step 3: Enhanced Sanskrit/Hindi corrections
    print("=== STEP 3: Enhanced Sanskrit/Hindi Corrections ===")
    try:
        sanskrit_corrections = processor._apply_enhanced_sanskrit_hindi_corrections(processed_segment.text)
        processed_segment.text = sanskrit_corrections['corrected_text']
        print(f"After Sanskrit corrections: {repr(processed_segment.text)}")
        # Check for corruption
        if "???" in processed_segment.text:
            print("🚨 CORRUPTION DETECTED: Question marks found in Sanskrit corrections step!")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Step 4: Legacy Sanskrit/Hindi corrections
    print("=== STEP 4: Legacy Lexicon Corrections ===")
    try:
        corrected_text, lexicon_corrections = processor._apply_lexicon_corrections(processed_segment.text)
        processed_segment.text = corrected_text
        print(f"After lexicon corrections: {repr(processed_segment.text)}")
        # Check for corruption
        if "???" in processed_segment.text:
            print("🚨 CORRUPTION DETECTED: Question marks found in lexicon corrections step!")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Step 5: NER processing
    print("=== STEP 5: NER Processing & Capitalization ===")
    if processor.enable_ner:
        try:
            # NER entity identification
            ner_result = processor.ner_model.identify_entities(processed_segment.text)
            print(f"Before NER capitalization: {repr(processed_segment.text)}")
            
            # Apply capitalization
            capitalization_result = processor.capitalization_engine.capitalize_text(processed_segment.text)
            processed_segment.text = capitalization_result.capitalized_text
            print(f"After NER capitalization: {repr(processed_segment.text)}")
            # Check for corruption
            if "???" in processed_segment.text:
                print("🚨 CORRUPTION DETECTED: Question marks found in NER processing step!")
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("NER processing disabled")
    print()
    
    # Step 6: Unicode normalization (our fix)
    print("=== STEP 6: Unicode Normalization (Fix 3) ===")
    print(f"Before Unicode normalization: {repr(processed_segment.text)}")
    normalized_text = processor._normalize_unicode_text(processed_segment.text)
    print(f"After Unicode normalization: {repr(normalized_text)}")
    
    if normalized_text != processed_segment.text:
        print("✅ Unicode normalization made changes")
    else:
        print("⚠️  Unicode normalization made no changes")
    
    print()
    print(f"FINAL RESULT: {repr(normalized_text)}")

def test_unicode_normalization_directly():
    """Test Unicode normalization method directly with various inputs."""
    print()
    print("=== DIRECT UNICODE NORMALIZATION TEST ===")
    
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    
    processor = SanskritPostProcessor()
    
    # Test cases with different types of corruption
    test_cases = [
        "Krishna",
        "K???a",  # Corrupted Krishna
        "dharma", 
        "Dharma",
        "Vishnu",
        "Vi??u",  # Corrupted Vishnu
        "Shiva",
        "?iva",   # Corrupted Shiva
    ]
    
    for test_input in test_cases:
        try:
            result = processor._normalize_unicode_text(test_input)
            print(f"Input:  {repr(test_input)}")
            print(f"Output: {repr(result)}")
            if result != test_input:
                print(f"✅ Changed: {test_input} → {result}")
            else:
                print(f"⚠️  No change")
            print()
        except Exception as e:
            print(f"Error processing {repr(test_input)}: {e}")
            print()

if __name__ == "__main__":
    debug_unicode_at_each_step()
    test_unicode_normalization_directly()