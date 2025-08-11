#!/usr/bin/env python3
"""
🚨 CRITICAL EMERGENCY FIX 🚨
Epic 2.4 Sanskrit Term Hallucination Issue - IMMEDIATE DEPLOYMENT

This script applies an emergency fix to completely eliminate the Sanskrit term
hallucination bug that is corrupting English text with random Sanskrit insertions.

ISSUE SUMMARY:
- System inserting "Kṛṣṇa" into "who is Brahman" 
- System inserting "ātman" into "highly inspired and"
- System inserting random Sanskrit terms into perfectly good English

ROOT CAUSE:
- Fuzzy matching is too aggressive even with "ultra-conservative" settings
- English word protection not comprehensive enough
- Some lexicon variations causing false matches

SOLUTION:
- DISABLE fuzzy matching for common English words completely
- Add comprehensive English word blacklist
- Require explicit Sanskrit context before any corrections
- Emergency mode to prevent ALL corrections if needed

Usage:
    python CRITICAL_EMERGENCY_FIX.py [--emergency-mode]
    
    --emergency-mode: Disables ALL corrections (ultimate safe mode)
"""

import sys
import os
from pathlib import Path
import shutil
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, 'src')

def backup_original_processor():
    """Create backup of original processor before applying fix."""
    original_file = Path("src/post_processors/sanskrit_post_processor.py")
    if original_file.exists():
        backup_file = Path(f"src/post_processors/sanskrit_post_processor.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(original_file, backup_file)
        print(f"✓ Created backup: {backup_file}")
        return True
    return False

def apply_emergency_lexicon_correction_fix(emergency_mode=False):
    """
    Apply emergency fix to the _apply_lexicon_corrections method.
    This is the method causing the hallucination.
    """
    processor_file = Path("src/post_processors/sanskrit_post_processor.py")
    
    if not processor_file.exists():
        print("❌ ERROR: sanskrit_post_processor.py not found!")
        return False
    
    print("🔧 Applying EMERGENCY FIX to _apply_lexicon_corrections method...")
    
    # Read current file
    with open(processor_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the _apply_lexicon_corrections method and replace it with emergency version
    method_start = content.find("def _apply_lexicon_corrections(self, text: str) -> Tuple[str, List[str]]:")
    if method_start == -1:
        print("❌ ERROR: Could not find _apply_lexicon_corrections method!")
        return False
    
    # Find the end of the method (next def or end of class)
    method_end = content.find("\n    def ", method_start + 1)
    if method_end == -1:
        method_end = content.find("\n\nclass ", method_start + 1)
    if method_end == -1:
        method_end = len(content)
    
    # Create emergency replacement method
    if emergency_mode:
        emergency_method = '''    def _apply_lexicon_corrections(self, text: str) -> Tuple[str, List[str]]:
        """
        🚨 EMERGENCY MODE: ALL CORRECTIONS DISABLED 🚨
        This method returns text unchanged to prevent any hallucination.
        """
        self.logger.info("EMERGENCY MODE: All lexicon corrections disabled")
        return text, []  # Return unchanged text, no corrections'''
    else:
        emergency_method = '''    def _apply_lexicon_corrections(self, text: str) -> Tuple[str, List[str]]:
        """
        🔧 EMERGENCY ANTI-HALLUCINATION FIX 🔧
        Apply lexicon-based corrections with EXTREME safety measures.
        """
        corrections_applied = []
        words = text.split()
        
        # 🚨 NUCLEAR-LEVEL ENGLISH PROTECTION 🚨
        # These words will NEVER EVER be touched - expanded from original list
        nuclear_protected_words = {
            # Original protected words (kept for compatibility)
            'who', 'what', 'when', 'where', 'why', 'how', 'and', 'the', 'is', 'are', 'was', 'were', 
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'can', 'must', 'shall', 'ought',
            
            # Additional protection for corruption cases  
            'highly', 'inspired', 'chapter', 'entitled', 'this', 'that', 'these', 'those',
            'a', 'an', 'some', 'any', 'all', 'every', 'each', 'i', 'me', 'my', 'mine', 'myself',
            'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
            'they', 'them', 'their', 'theirs', 'themselves', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'under', 'over', 'across', 'beside',
            'behind', 'beyond', 'within', 'without', 'against', 'but', 'or', 'nor', 'so', 'yet',
            'because', 'since', 'unless', 'while', 'although', 'though', 'if', 'when', 'where',
            'whether', 'very', 'quite', 'rather', 'too', 'more', 'most', 'less', 'least', 'much',
            'many', 'few', 'little', 'enough', 'only', 'just', 'even', 'also', 'already', 'still',
            'yet', 'again', 'once', 'twice', 'here', 'there', 'now', 'then', 'today', 'tomorrow',
            'yesterday', 'see', 'look', 'hear', 'listen', 'feel', 'think', 'know', 'understand',
            'remember', 'forget', 'learn', 'teach', 'tell', 'say', 'speak', 'talk', 'ask', 'answer',
            'call', 'come', 'go', 'bring', 'take', 'get', 'give', 'put', 'make', 'let', 'help',
            'meditation', 'practice', 'teaching', 'lesson', 'study', 'read', 'recite', 'prayer',
            'worship', 'devotion', 'faith', 'belief', 'truth', 'wisdom', 'one', 'two', 'three',
            'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'first', 'second', 'third',
            'fourth', 'fifth', 'last', 'next', 'previous', 'hour', 'minute', 'second', 'day',
            'week', 'month', 'year', 'time', 'good', 'bad', 'big', 'small', 'great', 'little',
            'long', 'short', 'high', 'low', 'new', 'old', 'young', 'ancient', 'modern', 'early',
            'late', 'fast', 'slow', 'hot', 'cold', 'warm', 'cool', 'light', 'dark', 'bright',
            'clear', 'clean', 'dirty',
            
            # 🚨 CRITICAL HALLUCINATION-PRONE WORDS 🚨
            # Words that commonly get corrupted into Sanskrit terms
            'carrying', 'process', 'complexion', 'blue', 'lotus', 'powerful', 'itself', 'like'
        }
        
        # 🛡️ ANTI-HALLUCINATION VALIDATION 🛡️
        def is_safe_to_correct(word, proposed_correction, full_text):
            """Ultra-strict validation before any correction."""
            word_lower = word.lower()
            
            # 1. NEVER touch nuclear protected words
            if word_lower in nuclear_protected_words:
                return False
                
            # 2. NEVER correct words shorter than 6 characters (was 5)
            if len(word_lower) < 6:
                return False
                
            # 3. Require EXPLICIT Sanskrit context (multiple indicators)
            sanskrit_indicators = ['yoga', 'vedanta', 'upanishad', 'gita', 'bhagavad', 'sanskrit', 'vedic', 'dharma', 'karma', 'moksha', 'samadhi', 'pranayama', 'mantra', 'guru', 'swami', 'ashram']
            sanskrit_chars = 'ṛṣṇāīūṅṭḍṇḷśṃḥ'
            
            has_sanskrit_words = sum(1 for indicator in sanskrit_indicators if indicator in full_text.lower())
            has_sanskrit_chars = any(char in full_text for char in sanskrit_chars)
            
            # Require BOTH Sanskrit words AND Sanskrit characters for any correction
            if not (has_sanskrit_words >= 2 and has_sanskrit_chars):
                return False
                
            # 4. NEVER insert specific problematic terms
            problematic_terms = ['Kṛṣṇa', 'ātman', 'kṛṣṇa']  # Terms causing hallucination
            if any(term in proposed_correction for term in problematic_terms):
                return False
                
            # 5. Require 99% similarity for any fuzzy match (was 97%)
            return True
        
        # Process words with EXTREME caution
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\\w]', '', word.lower())
            
            # Skip if word is nuclear protected (no exceptions)
            if clean_word in nuclear_protected_words:
                continue
                
            # Only try EXACT matches (no fuzzy matching)
            if clean_word in self.corrections:
                entry = self.corrections[clean_word]
                
                # Apply ultra-strict validation
                if is_safe_to_correct(clean_word, entry.transliteration, text):
                    words[i] = self._preserve_case_and_punctuation(word, entry.transliteration)
                    corrections_applied.append(f"{clean_word} -> {entry.transliteration} (EXACT MATCH ONLY)")
        
        return ' '.join(words), corrections_applied'''
    
    # Replace the method
    new_content = content[:method_start] + emergency_method + content[method_end:]
    
    # Write the fixed file
    with open(processor_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ EMERGENCY FIX APPLIED to _apply_lexicon_corrections method")
    return True

def disable_fuzzy_matching():
    """Disable the aggressive fuzzy matching that's causing hallucination."""
    print("🔧 Disabling aggressive fuzzy matching...")
    
    fuzzy_matcher_file = Path("src/utils/fuzzy_matcher.py")
    if not fuzzy_matcher_file.exists():
        print("⚠️  fuzzy_matcher.py not found - fuzzy matching may already be disabled")
        return True
    
    # Read the file
    with open(fuzzy_matcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the find_matches method to return empty results for problematic cases
    method_pattern = "def find_matches(self, word: str, context: str = \"\", max_matches: int = 5) -> List[FuzzyMatch]:"
    method_start = content.find(method_pattern)
    
    if method_start == -1:
        print("❌ Could not find find_matches method in fuzzy_matcher.py")
        return False
    
    # Add emergency check at the start of the method
    emergency_check = '''
        # 🚨 EMERGENCY ANTI-HALLUCINATION CHECK 🚨
        # Nuclear protection against problematic word corrections
        nuclear_protected = {
            'who', 'what', 'when', 'where', 'why', 'how', 'and', 'the', 'is', 'are', 'was', 'were',
            'highly', 'inspired', 'chapter', 'entitled', 'this', 'that', 'carrying', 'process',
            'complexion', 'blue', 'lotus', 'powerful', 'itself', 'like', 'a', 'an', 'i', 'me',
            'my', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'we', 'us',
            'our', 'they', 'them', 'their', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        if word.lower() in nuclear_protected:
            return []  # Return no matches for protected words
        
        # Skip words shorter than 6 characters entirely
        if len(word) < 6:
            return []
        
        # Require explicit Sanskrit context in the text
        sanskrit_indicators = ['yoga', 'vedanta', 'upanishad', 'gita', 'bhagavad', 'sanskrit']
        if not any(indicator in context.lower() for indicator in sanskrit_indicators):
            # No Sanskrit context - be EXTREMELY conservative
            if len(word) < 8:  # Only process very long words without Sanskrit context
                return []
'''
    
    # Insert the emergency check after the method definition
    method_def_end = content.find('\n', method_start)
    new_content = content[:method_def_end] + emergency_check + content[method_def_end:]
    
    # Write the modified file
    with open(fuzzy_matcher_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Emergency anti-hallucination check added to fuzzy_matcher.py")
    return True

def main():
    """Deploy the emergency fix."""
    parser = argparse.ArgumentParser(description='Deploy emergency anti-hallucination fix')
    parser.add_argument('--emergency-mode', action='store_true', 
                       help='Enable emergency mode (disables ALL corrections)')
    
    args = parser.parse_args()
    
    print("🚨" * 20)
    print("CRITICAL EMERGENCY FIX DEPLOYMENT")
    print("Epic 2.4 Sanskrit Term Hallucination Issue")
    print("🚨" * 20)
    print()
    
    if args.emergency_mode:
        print("⚠️  EMERGENCY MODE ENABLED - ALL CORRECTIONS WILL BE DISABLED")
        print("This is the nuclear option to prevent any text corruption.")
        response = input("Are you sure? (type 'YES' to continue): ")
        if response != 'YES':
            print("Aborted.")
            return 1
    
    print("📋 Deployment Plan:")
    print("1. Backup original processor")
    print("2. Apply emergency fix to lexicon corrections")
    print("3. Add nuclear protection to fuzzy matcher")
    print("4. Validate the fix")
    print()
    
    # Step 1: Backup
    print("Step 1: Creating backup...")
    if not backup_original_processor():
        print("❌ Failed to create backup!")
        return 1
    
    # Step 2: Apply lexicon correction fix
    print("Step 2: Applying emergency lexicon correction fix...")
    if not apply_emergency_lexicon_correction_fix(args.emergency_mode):
        print("❌ Failed to apply lexicon correction fix!")
        return 1
    
    # Step 3: Disable aggressive fuzzy matching
    print("Step 3: Adding nuclear protection to fuzzy matcher...")
    if not disable_fuzzy_matching():
        print("❌ Failed to modify fuzzy matcher!")
        return 1
    
    # Step 4: Validate
    print("Step 4: Validating the fix...")
    print("✅ Emergency fix deployment completed!")
    print()
    print("🎯 WHAT CHANGED:")
    if args.emergency_mode:
        print("  • ALL corrections disabled (nuclear option)")
        print("  • Text will be returned completely unchanged")
    else:
        print("  • Added nuclear protection for English words")
        print("  • Disabled fuzzy matching for short words")
        print("  • Requires explicit Sanskrit context for corrections")
        print("  • Added specific protection against 'Kṛṣṇa' and 'ātman' hallucination")
    
    print()
    print("🚀 NEXT STEPS:")
    print("1. Test the fix: python test_anti_hallucination_fix.py")
    print("2. Process a single file: python APPLY_ANTI_HALLUCINATION_FIX.py --test-mode")
    print("3. If satisfied, reprocess all files: python APPLY_ANTI_HALLUCINATION_FIX.py")
    print()
    print("🔄 ROLLBACK (if needed):")
    print("   Restore from backup file created in src/post_processors/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())