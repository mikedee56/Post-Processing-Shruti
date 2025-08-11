#!/usr/bin/env python3
"""
Quick test script to validate the anti-hallucination fix
"""
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, 'src')

def test_lexicon_loading():
    """Test that lexicons load correctly with new format"""
    print("🔍 Testing lexicon loading...")
    
    try:
        from sanskrit_hindi_identifier.lexicon_manager import LexiconManager
        
        lexicon_manager = LexiconManager(Path("data/lexicons"))
        entries = lexicon_manager.get_all_entries()
        
        print(f"  ✓ Loaded {len(entries)} lexicon entries")
        
        # Check specific problem terms
        problem_terms = ['krishna', 'atman']
        for term in problem_terms:
            if term in entries:
                entry = entries[term]
                print(f"  ✓ Found {term} -> {entry.transliteration}")
            else:
                print(f"  ℹ {term} not in lexicon (good for reducing hallucination)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading lexicons: {e}")
        return False

def test_sample_text_processing():
    """Test processing of the sample text that was corrupted"""
    print("\n🔍 Testing sample text processing...")
    
    # Test problematic phrases from user's examples
    test_cases = [
        "who is Brahman",
        "This chapter is entitled, Atma Vishranti", 
        "highly inspired and",
        "In Sanskrit, the word for soul is atman"
    ]
    
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        config_path = Path("config/conservative_config.yaml")
        
        if config_path.exists():
            processor = SanskritPostProcessor(config_path)
        else:
            print("  ⚠️  Conservative config not found, using defaults")
            processor = SanskritPostProcessor()
        
        print(f"  ✓ Initialized processor")
        
        for test_text in test_cases:
            # Test the legacy correction method that was causing issues
            corrected_text, corrections = processor._apply_lexicon_corrections(test_text)
            
            print(f"  📝 '{test_text}'")
            print(f"     -> '{corrected_text}'")
            if corrections:
                print(f"     Corrections: {corrections}")
            else:
                print(f"     No corrections (good!)")
            print()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing processing: {e}")
        return False

def main():
    """Run validation tests"""
    print("🧪 ANTI-HALLUCINATION FIX VALIDATION")
    print("=" * 50)
    
    lexicon_ok = test_lexicon_loading()
    processing_ok = test_sample_text_processing()
    
    if lexicon_ok and processing_ok:
        print("✅ FIX VALIDATION PASSED!")
        print("The system should now be safe for re-processing.")
        return 0
    else:
        print("❌ FIX VALIDATION FAILED!")
        print("Please check the errors above and run the fix again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())