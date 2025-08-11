#!/usr/bin/env python3
"""
EMERGENCY STOP: Eliminate Sanskrit Term Hallucination
"""
import sys
from pathlib import Path
import yaml

def create_conservative_lexicons():
    """Create ultra-conservative lexicons to stop hallucination"""
    
    lexicon_dir = Path("data/lexicons")
    
    print("=== STOPPING SANSKRIT HALLUCINATION ===")
    print()
    
    # Ultra-conservative corrections - only obvious, unambiguous cases
    corrections = {
        "bhagavad": {
            "variations": ["bhagvad", "bhagwad"],
            "transliteration": "Bhagavad",
            "confidence": 0.95
        },
        "gita": {
            "variations": ["geeta", "geetha"], 
            "transliteration": "Gītā",
            "confidence": 0.95
        }
        # REMOVED: krishna, atman, and other problematic terms
    }
    
    # Only clear proper nouns with exact matches
    proper_nouns = {
        "Arjuna": {
            "variations": ["arjun"],
            "confidence": 0.98
        }
        # REMOVED: Krishna variations that cause hallucination
    }
    
    # Minimal phrase cleanup only
    phrases = {
        " um,": ",",
        " uh,": ",", 
        ", you know": ""
    }
    
    # NO verse substitutions
    verses = {}
    
    # Write conservative lexicons
    with open(lexicon_dir / "corrections.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(corrections, f, allow_unicode=True, default_flow_style=False)
    
    with open(lexicon_dir / "proper_nouns.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(proper_nouns, f, allow_unicode=True, default_flow_style=False)
        
    with open(lexicon_dir / "phrases.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(phrases, f, allow_unicode=True, default_flow_style=False)
    
    with open(lexicon_dir / "verses.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(verses, f, allow_unicode=True, default_flow_style=False)
    
    print("✓ Created ultra-conservative lexicons")
    print("✓ Removed problematic Krishna/atman entries")
    print("✓ Kept only high-confidence, obvious corrections")

def test_no_hallucination():
    """Test that hallucination examples are fixed"""
    
    print("\n=== TESTING HALLUCINATION FIX ===")
    
    test_cases = [
        "who is Brahman",
        "This chapter is entitled, Atma Vishranti", 
        "highly inspired and",
        "are coming very close"
    ]
    
    print("Test cases that should NOT be corrupted:")
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. '{case}' → Should remain unchanged")
    
    print("\nAfter re-processing, these should be identical to originals")
    print("No random Krishna/atman insertions allowed!")

if __name__ == "__main__":
    # Backup current lexicons first
    import shutil
    from datetime import datetime
    
    backup_dir = Path(f"data/lexicons_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if Path("data/lexicons").exists():
        shutil.copytree("data/lexicons", backup_dir)
        print(f"✓ Backed up current lexicons to {backup_dir}")
    
    create_conservative_lexicons()
    test_no_hallucination()
    
    print("\n=== NEXT STEPS ===")
    print("1. Run: py -3.10 simple_batch.py")
    print("2. Check that NO random Sanskrit terms are inserted") 
    print("3. Verify your problematic examples are fixed")
    print("4. If still issues, we go even more conservative")