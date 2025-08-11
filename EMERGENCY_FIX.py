#!/usr/bin/env python3
"""
EMERGENCY FIX: Disable aggressive corrections that are causing hallucinations
"""
import sys
from pathlib import Path
import shutil
import datetime

def backup_and_fix_system():
    """Backup current lexicons and reduce aggressiveness"""
    
    print("=== EMERGENCY FIX: Reducing System Aggressiveness ===")
    print()
    
    lexicon_dir = Path("data/lexicons")
    backup_dir = Path("data/lexicons_backup_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Backup current lexicons
    if lexicon_dir.exists():
        shutil.copytree(lexicon_dir, backup_dir)
        print(f"✓ Backed up current lexicons to: {backup_dir}")
    
    # Create minimal, conservative lexicons
    create_minimal_lexicons(lexicon_dir)
    print("✓ Created conservative lexicon files")
    
    print()
    print("=== FIXED ISSUES ===")
    print("• Removed aggressive fuzzy matching")
    print("• Disabled random Sanskrit term insertion")
    print("• Kept only high-confidence corrections")
    print("• Preserved IAST transliteration for actual Sanskrit")
    print()
    print("=== TO RE-PROCESS ===")
    print("Run: py -3.10 simple_batch.py")
    print("This should now be much more conservative")

def create_minimal_lexicons(lexicon_dir):
    """Create minimal, conservative lexicon files"""
    
    lexicon_dir.mkdir(exist_ok=True)
    
    # Very conservative corrections - only obvious mistakes
    corrections = {
        "gita": {
            "variations": ["geeta", "geetha"],
            "transliteration": "gītā",
            "is_proper_noun": True,
            "confidence": 0.95
        },
        "yoga": {
            "variations": ["yog"],
            "transliteration": "yoga",
            "is_proper_noun": False,
            "confidence": 0.95
        },
        "dharma": {
            "variations": ["dharama"],
            "transliteration": "dharma", 
            "is_proper_noun": False,
            "confidence": 0.95
        }
    }
    
    # Only clear proper nouns
    proper_nouns = {
        "Krishna": {
            "variations": ["krishna", "krsna"],
            "transliteration": "Kṛṣṇa",
            "is_proper_noun": True,
            "confidence": 0.98
        },
        "Bhagavad": {
            "variations": ["bhagvad", "bhagwad"],
            "transliteration": "Bhagavad",
            "is_proper_noun": True,
            "confidence": 0.98
        }
    }
    
    # Minimal phrases - only obvious cleanup
    phrases = {
        "um,": "",
        "uh,": "",
        ", you know": "",
        " you know,": ","
    }
    
    # No verse substitutions for now
    verses = {}
    
    # Write files
    import yaml
    
    with open(lexicon_dir / "corrections.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(corrections, f, allow_unicode=True, default_flow_style=False)
    
    with open(lexicon_dir / "proper_nouns.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(proper_nouns, f, allow_unicode=True, default_flow_style=False)
        
    with open(lexicon_dir / "phrases.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(phrases, f, allow_unicode=True, default_flow_style=False)
    
    with open(lexicon_dir / "verses.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(verses, f, allow_unicode=True, default_flow_style=False)

def restore_originals():
    """Restore original SRT files from raw_srts"""
    
    print("=== RESTORING ORIGINAL FILES ===")
    print()
    
    raw_dir = Path("data/raw_srts") 
    processed_dir = Path("data/processed_srts")
    
    # Remove corrupted enhanced files
    corrupted_files = list(processed_dir.glob("*_enhanced.srt"))
    for file in corrupted_files:
        file.unlink()
        print(f"✓ Removed corrupted: {file.name}")
    
    print(f"✓ Cleared {len(corrupted_files)} corrupted enhanced files")
    print("✓ Your originals in data/raw_srts/ are safe")
    print()
    print("Ready to re-process with fixed settings")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_originals()
    else:
        backup_and_fix_system()
        restore_originals()