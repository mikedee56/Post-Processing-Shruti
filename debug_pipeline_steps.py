#!/usr/bin/env python3
"""
CRITICAL DIAGNOSTIC: Pipeline Step-by-Step Analysis
Traces exactly where each failure occurs in the integration pipeline
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Suppress verbose logging
logging.getLogger().setLevel(logging.ERROR)

def debug_pipeline_steps():
    """Debug each step of the processing pipeline to identify failure points."""
    print("=== PIPELINE STEP-BY-STEP DIAGNOSTIC ===")
    print()
    
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    from utils.srt_parser import SRTSegment
    
    processor = SanskritPostProcessor()
    
    # Create test segment
    test_text = "today we study krishna in chapter two verse twenty five"
    segment = SRTSegment(
        index=1,
        start_time="00:00:01,000",
        end_time="00:00:05,000", 
        text=test_text,
        raw_text=test_text
    )
    
    print(f"ORIGINAL: {test_text}")
    print()
    
    # Create copy to track changes
    import copy
    processed_segment = copy.deepcopy(segment)
    
    # Step 1: Enhanced Text Normalization
    print("=== STEP 1: Enhanced Text Normalization ===")
    if hasattr(processor.text_normalizer, 'normalize_with_advanced_tracking'):
        advanced_result = processor.text_normalizer.normalize_with_advanced_tracking(processed_segment.text)
        processed_segment.text = advanced_result.corrected_text
        print(f"Result: {processed_segment.text}")
        print(f"Changes: {advanced_result.corrections_applied}")
    else:
        print("Advanced normalization not available")
    print()
    
    # Step 2: Enhanced contextual number processing
    print("=== STEP 2: Contextual Number Processing ===")
    if hasattr(processor, 'number_processor'):
        number_result = processor.number_processor.process_numbers(processed_segment.text, context="spiritual")
        processed_segment.text = number_result.processed_text
        print(f"Result: {processed_segment.text}")
        print(f"Conversions: {len(number_result.conversions)}")
        for conv in number_result.conversions:
            print(f"  - {conv.original_text} -> {conv.converted_text}")
    else:
        print("Number processor not available")
    print()
    
    # Step 3: Enhanced Sanskrit/Hindi corrections
    print("=== STEP 3: Enhanced Sanskrit/Hindi Corrections ===")
    try:
        sanskrit_corrections = processor._apply_enhanced_sanskrit_hindi_corrections(processed_segment.text)
        processed_segment.text = sanskrit_corrections['corrected_text']
        print(f"Result: {processed_segment.text}")
        print(f"Corrections: {len(sanskrit_corrections.get('corrections_applied', []))}")
        for correction in sanskrit_corrections.get('corrections_applied', []):
            print(f"  - {correction.original_text} -> {correction.corrected_text}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Step 4: Legacy Sanskrit/Hindi corrections
    print("=== STEP 4: Legacy Lexicon Corrections ===")
    try:
        corrected_text, lexicon_corrections = processor._apply_lexicon_corrections(processed_segment.text)
        processed_segment.text = corrected_text
        print(f"Result: {processed_segment.text}")
        print(f"Lexicon corrections: {len(lexicon_corrections)}")
        for correction in lexicon_corrections:
            print(f"  - {correction}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Step 5: NER processing
    print("=== STEP 5: NER Processing & Capitalization ===")
    if processor.enable_ner:
        try:
            # NER entity identification
            ner_result = processor.ner_model.identify_entities(processed_segment.text)
            print(f"NER entities: {len(ner_result.entities)}")
            for entity in ner_result.entities:
                print(f"  - {entity.text} ({entity.category.value}) conf={entity.confidence:.3f}")
            
            # Apply capitalization
            capitalization_result = processor.capitalization_engine.capitalize_text(processed_segment.text)
            processed_segment.text = capitalization_result.capitalized_text
            print(f"Result: {processed_segment.text}")
            print(f"Capitalizations: {len(capitalization_result.changes_made)}")
            for change in capitalization_result.changes_made:
                print(f"  - {change}")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("NER processing disabled")
    print()
    
    print(f"FINAL RESULT: {processed_segment.text}")
    print()
    
    # Validate against expected
    expected = "Today we study Krishna in Chapter 2 verse 25."
    issues = []
    
    if "Chapter 2 verse 25" not in processed_segment.text:
        if "Chapter 2 Verse 25" in processed_segment.text:
            issues.append("ISSUE: 'verse' capitalized incorrectly")
        else:
            issues.append("ISSUE: Scriptural conversion failed")
    
    if "Krishna" not in processed_segment.text:
        if "K" in processed_segment.text and "a" in processed_segment.text:
            issues.append("ISSUE: Krishna has Unicode corruption")
        else:
            issues.append("ISSUE: Krishna capitalization failed")
    
    if issues:
        print("IDENTIFIED ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No issues found")

def debug_idiomatic_preservation():
    """Debug idiomatic preservation specifically."""
    print()
    print("=== IDIOMATIC PRESERVATION DIAGNOSTIC ===")
    print()
    
    from post_processors.sanskrit_post_processor import SanskritPostProcessor
    from utils.srt_parser import SRTSegment
    
    processor = SanskritPostProcessor()
    
    # Test idiomatic text
    test_text = "and one by one, the students learned about dharma"
    segment = SRTSegment(
        index=2,
        start_time="00:00:06,000",
        end_time="00:00:10,000", 
        text=test_text,
        raw_text=test_text
    )
    
    print(f"ORIGINAL: {test_text}")
    print()
    
    # Test advanced text normalizer directly
    if hasattr(processor.text_normalizer, 'convert_numbers_with_context'):
        direct_result = processor.text_normalizer.convert_numbers_with_context(test_text)
        print(f"Direct normalization: {direct_result}")
    
    # Process through full pipeline
    file_metrics = processor.metrics_collector.create_file_metrics('debug_test')
    processed_segment = processor._process_srt_segment(segment, file_metrics)
    
    print(f"Full pipeline result: {processed_segment.text}")
    
    if "one by one" in processed_segment.text:
        print("✅ Idiomatic preservation: SUCCESS")
    else:
        print("❌ Idiomatic preservation: FAILED")

if __name__ == "__main__":
    debug_pipeline_steps()
    debug_idiomatic_preservation()