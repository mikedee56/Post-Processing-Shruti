#!/usr/bin/env python3
"""
Epic 2.4 System Tuning Guide & Configuration Helper
"""
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def show_tuning_options():
    """Show what can be tuned in Epic 2.4"""
    
    print("=== Epic 2.4 Tuning Options ===")
    print()
    
    print("1. CONFIDENCE THRESHOLDS")
    print("   • Fuzzy matching confidence (default: 0.8)")
    print("   • IAST correction confidence (default: 0.8)")
    print("   • Overall processing confidence")
    print()
    
    print("2. LEXICON MANAGEMENT")
    print("   • Add custom Sanskrit/Hindi terms")
    print("   • Update proper noun capitalizations") 
    print("   • Modify verse corrections")
    print("   • Add phrase replacements")
    print()
    
    print("3. PROCESSING BEHAVIOR")
    print("   • Enable/disable sandhi preprocessing")
    print("   • Adjust filler word removal aggressiveness")
    print("   • Control IAST transliteration strictness")
    print("   • Modify number conversion rules")
    print()
    
    print("4. PERFORMANCE SETTINGS")
    print("   • Batch processing size")
    print("   • Memory usage limits")
    print("   • Concurrent processing")
    print()

def check_current_config():
    """Check current system configuration"""
    
    print("=== Current System Configuration ===")
    print()
    
    # Check lexicon files
    lexicon_dir = Path("data/lexicons")
    lexicon_files = ["corrections.yaml", "proper_nouns.yaml", "phrases.yaml", "verses.yaml"]
    
    print("LEXICON STATUS:")
    total_entries = 0
    for lex_file in lexicon_files:
        file_path = lexicon_dir / lex_file
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            entries = len(data) if data else 0
            total_entries += entries
            print(f"  {lex_file:<20} {entries:>3} entries")
        else:
            print(f"  {lex_file:<20}   MISSING")
    
    print(f"  Total lexicon entries: {total_entries}")
    print()
    
    # Check configuration files
    config_dir = Path("config")
    config_files = ["contextual_config.yaml", "scripture_config.yaml"]
    
    print("CONFIG FILES:")
    for config_file in config_files:
        file_path = config_dir / config_file
        status = "EXISTS" if file_path.exists() else "MISSING"
        print(f"  {config_file:<25} {status}")
    
    print()

def suggest_improvements(evaluation_file=None):
    """Suggest improvements based on evaluation results"""
    
    print("=== Suggested Improvements ===")
    print()
    
    if evaluation_file and Path(evaluation_file).exists():
        import json
        with open(evaluation_file, 'r') as f:
            eval_data = json.load(f)
        
        success_rate = eval_data.get("success_rate", 0)
        quality_indicators = eval_data.get("quality_indicators", {})
        
        print("Based on your latest evaluation:")
        print()
        
        if success_rate < 0.95:
            print(f"• SUCCESS RATE: {success_rate:.1%} - Consider:")
            print("  - Check file encodings (should be UTF-8)")
            print("  - Review failed files for format issues")
            print("  - Lower confidence thresholds if too strict")
            print()
        
        if not quality_indicators.get("IAST transliteration", True):
            print("• IAST TRANSLITERATION missing - Consider:")
            print("  - Adding more Sanskrit terms to lexicon")
            print("  - Checking if your content contains Sanskrit words")
            print("  - Verifying IAST transliterator is enabled")
            print()
        
        if not quality_indicators.get("Clean text", True):
            print("• FILLER WORDS still present - Consider:")
            print("  - Strengthening conversational cleanup rules")
            print("  - Adding custom filler patterns")
            print("  - Review your specific content patterns")
            print()
    
    else:
        print("General tuning recommendations:")
        print()
        print("• RUN EVALUATION FIRST:")
        print("  py -3.10 EVALUATE_OUTPUT.py")
        print()
        print("• FOR BETTER SANSKRIT RECOGNITION:")
        print("  - Add your specific terms to data/lexicons/corrections.yaml")
        print("  - Check proper_nouns.yaml for speaker names, places")
        print()
        print("• FOR BETTER FILLER REMOVAL:")
        print("  - Review data/lexicons/phrases.yaml")
        print("  - Add your speaker's specific patterns")
        print()

def add_custom_term():
    """Interactive helper to add custom Sanskrit terms"""
    
    print("=== Add Custom Sanskrit Term ===")
    print()
    
    print("This will help you add a new Sanskrit/Hindi term to the lexicon.")
    print("Example: If ASR transcribes 'karma' as 'carma', you can add this correction.")
    print()
    
    try:
        original_term = input("Enter the correct term (e.g., 'karma'): ").strip()
        if not original_term:
            print("No term entered. Cancelled.")
            return
        
        variations = input("Enter common misspellings (comma-separated, e.g., 'carma,karuma'): ").strip()
        if not variations:
            print("No variations entered. Cancelled.")
            return
        
        is_proper = input("Is this a proper noun? (y/n): ").strip().lower() == 'y'
        transliteration = input(f"IAST transliteration (leave blank if same as '{original_term}'): ").strip()
        
        if not transliteration:
            transliteration = original_term
        
        print()
        print("Summary:")
        print(f"  Term: {original_term}")
        print(f"  Variations: {variations}")
        print(f"  Proper noun: {is_proper}")
        print(f"  IAST: {transliteration}")
        print()
        
        confirm = input("Add this term? (y/n): ").strip().lower()
        if confirm == 'y':
            # Add to appropriate lexicon file
            lexicon_file = "proper_nouns.yaml" if is_proper else "corrections.yaml"
            print(f"Add this manually to data/lexicons/{lexicon_file}")
            print(f"Or use the lexicon management tools in Epic 2.4")
        else:
            print("Cancelled.")
    
    except KeyboardInterrupt:
        print("\nCancelled.")

def main():
    """Main tuning interface"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "config":
            check_current_config()
        elif command == "improve":
            suggest_improvements()
        elif command == "add":
            add_custom_term()
        else:
            print(f"Unknown command: {command}")
            print("Usage: py -3.10 TUNE_SYSTEM.py [config|improve|add]")
    else:
        show_tuning_options()
        print()
        print("Commands:")
        print("  py -3.10 TUNE_SYSTEM.py config    - Show current configuration")
        print("  py -3.10 TUNE_SYSTEM.py improve   - Get improvement suggestions")
        print("  py -3.10 TUNE_SYSTEM.py add       - Add custom Sanskrit term")

if __name__ == "__main__":
    main()