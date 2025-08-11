# Current TextNormalizer Implementation - PROBLEMATIC CODE
# Source: src/utils/text_normalizer.py

"""
CRITICAL ISSUE: This file contains the problematic number conversion logic
that converts "one by one" to "1 by 1" causing quality degradation.

The issue is in the convert_numbers() method which uses simple find-and-replace
without any context awareness for idiomatic expressions.
"""

import re
from typing import Dict, List

class TextNormalizer:
    def __init__(self, config=None):
        self.config = config or {}
        self._setup_number_mappings()
    
    def _setup_number_mappings(self):
        """Set up number word to digit mappings."""
        # PROBLEMATIC: These basic mappings cause the issues
        self.basic_numbers = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        
        # Compound numbers (also problematic in context)
        self.compound_numbers = {
            "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
            "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
            "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
            "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
            "ninety": "90"
        }
    
    def convert_numbers(self, text: str) -> str:
        """
        Convert number words to digits.
        
        CRITICAL PROBLEM: This method blindly converts ALL number words
        without considering context. It treats mathematical quantities
        and idiomatic expressions identically.
        
        Examples of problems:
        - "one by one" → "1 by 1" (should preserve idiom)
        - "step by step" → "step by step" (currently OK, but fragile)
        - "all the six children" → "all the 6 children" (poor style)
        """
        
        # PROBLEMATIC LOGIC: No context awareness
        for word_num, digit in self.basic_numbers.items():
            pattern = rf'\b{re.escape(word_num)}\b'
            # This line causes "one by one" → "1 by 1"
            text = re.sub(pattern, digit, text, flags=re.IGNORECASE)
        
        # Compound numbers have same problem
        for word_num, digit in self.compound_numbers.items():
            pattern = rf'\b{re.escape(word_num)}\b'
            text = re.sub(pattern, digit, text, flags=re.IGNORECASE)
        
        return text
    
    def _convert_compound_numbers(self, text: str) -> str:
        """Convert compound numbers like 'twenty five' to '25'."""
        # This logic works for mathematical contexts but fails for narrative
        compound_pattern = r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\b'
        
        def replace_compound(match):
            tens = match.group(1)
            units = match.group(2)
            tens_digit = self.compound_numbers.get(tens.lower(), tens)
            units_digit = self.basic_numbers.get(units.lower(), units)
            return str(int(tens_digit) + int(units_digit))
        
        return re.sub(compound_pattern, replace_compound, text, flags=re.IGNORECASE)

    def normalize_with_advanced_tracking(self, text):
        """
        Main normalization method called by SanskritPostProcessor.
        
        This calls convert_numbers() which causes our quality issues.
        """
        original_text = text
        changes_applied = []
        
        # Step 1: Remove filler words (works fine)
        text = self.remove_filler_words(text)
        if text != original_text:
            changes_applied.append("filler_removal")
        
        # Step 2: PROBLEMATIC - Number conversion without context
        before_numbers = text
        text = self.convert_numbers(text)  # THIS CAUSES THE ISSUES
        if text != before_numbers:
            changes_applied.append("number_conversion")
        
        # Step 3: Other normalizations (work fine)
        text = self.normalize_punctuation(text)
        text = self.capitalize_sentences(text)
        
        return AdvancedNormalizationResult(
            original_text=original_text,
            corrected_text=text,
            corrections_applied=changes_applied,
            processing_time=0.001  # Placeholder
        )

# What we need instead: Context-aware processing
"""
PROPOSED SOLUTION: Replace convert_numbers() with intelligent context analysis

def convert_numbers_with_context(self, text: str) -> str:
    # Use MCP to classify each number in context
    tokens = self.mcp_client.tokenize(text)
    
    for token in tokens:
        if token.is_number_word:
            context = self.mcp_client.classify_context(token, surrounding_text)
            
            if context == "idiomatic":
                # Preserve "one by one", "step by step", etc.
                continue
            elif context == "mathematical":
                # Convert "two thousand five" → "2005"
                text = text.replace(token.text, convert_to_digit(token.text))
            elif context == "narrative":
                # Handle "all the six children" based on style preferences
                text = apply_narrative_rules(token, text)
    
    return text
"""

# Integration point in SanskritPostProcessor
"""
File: src/post_processors/sanskrit_post_processor.py
Method: _process_srt_segment() around line 460

Current problematic flow:
    normalized_result = self.text_normalizer.normalize_with_advanced_tracking(text)
    # This calls convert_numbers() which causes quality issues

Enhanced flow needed:
    if self.config.get('enable_advanced_normalization', False):
        normalized_result = self.advanced_normalizer.normalize_with_context(text)
    else:
        normalized_result = self.text_normalizer.normalize_with_advanced_tracking(text)
"""