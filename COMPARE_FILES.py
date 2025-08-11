#!/usr/bin/env python3
"""
Compare Original vs Enhanced SRT Files - Side by Side
"""
import sys
from pathlib import Path

def compare_side_by_side():
    """Show detailed before/after comparison"""
    
    raw_dir = Path("data/raw_srts")
    processed_dir = Path("data/processed_srts")
    
    print("=== Side-by-Side File Comparison ===")
    print()
    
    # Find first enhanced file for demo
    enhanced_files = list(processed_dir.glob("*_enhanced.srt"))
    if not enhanced_files:
        print("No enhanced files found. Run processing first.")
        return
    
    enhanced_file = enhanced_files[0]
    original_name = enhanced_file.name.replace("_enhanced.srt", ".srt")
    original_file = raw_dir / original_name
    
    if not original_file.exists():
        print(f"Original file not found: {original_file}")
        return
    
    print(f"Comparing: {original_name}")
    print("-" * 80)
    
    # Read files
    with open(original_file, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
    
    with open(enhanced_file, 'r', encoding='utf-8') as f:
        enhanced_lines = f.readlines()
    
    # Show first few segments for comparison
    segments_shown = 0
    i = 0
    
    while i < min(len(original_lines), len(enhanced_lines)) and segments_shown < 3:
        orig_line = original_lines[i].strip()
        enhanced_line = enhanced_lines[i].strip()
        
        # Look for text content (not timestamps or numbers)
        if orig_line and not orig_line.isdigit() and "-->" not in orig_line:
            if orig_line != enhanced_line:  # Only show if different
                print(f"SEGMENT {segments_shown + 1}:")
                print(f"  BEFORE: {orig_line}")
                print(f"  AFTER:  {enhanced_line}")
                
                # Highlight specific changes
                changes = []
                if "ā" in enhanced_line or "ī" in enhanced_line or "ū" in enhanced_line:
                    changes.append("IAST added")
                if " um," in orig_line and " um," not in enhanced_line:
                    changes.append("Filler removed")
                if " uh," in orig_line and " uh," not in enhanced_line:
                    changes.append("Filler removed")
                if any(c.isdigit() for c in enhanced_line) and len([c for c in enhanced_line if c.isdigit()]) > len([c for c in orig_line if c.isdigit()]):
                    changes.append("Numbers converted")
                
                if changes:
                    print(f"  CHANGES: {', '.join(changes)}")
                print()
                segments_shown += 1
        
        i += 1
    
    print(f"Comparison complete. Checked first {segments_shown} different segments.")
    print()
    print("To see more files, run: py -3.10 COMPARE_FILES.py all")

def compare_all_files():
    """Quick comparison of all files"""
    raw_dir = Path("data/raw_srts")
    processed_dir = Path("data/processed_srts")
    
    print("=== All Files Quality Overview ===")
    print()
    
    enhanced_files = list(processed_dir.glob("*_enhanced.srt"))
    
    for enhanced_file in enhanced_files:
        original_name = enhanced_file.name.replace("_enhanced.srt", ".srt")
        original_file = raw_dir / original_name
        
        if original_file.exists():
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(enhanced_file, 'r', encoding='utf-8') as f:
                enhanced = f.read()
            
            # Quick quality check
            quality_score = 0
            indicators = []
            
            if any(char in enhanced for char in "āīūṛḷēōṃḥ"):
                quality_score += 1
                indicators.append("IAST")
            
            if original.count(" um,") > enhanced.count(" um,"):
                quality_score += 1
                indicators.append("Filler-")
                
            if original.count(" uh,") > enhanced.count(" uh,"):
                quality_score += 1
                indicators.append("Cleanup")
            
            changes = len(original) != len(enhanced)
            
            print(f"{original_name:<30} Quality: {quality_score}/3  {' '.join(indicators):<20} {'Changed' if changes else 'Unchanged'}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        compare_all_files()
    else:
        compare_side_by_side()