"""
Story 4.1: Advanced Text Normalizer Enhancement Patch
Fixes critical issues and integrates with MCP Infrastructure Foundation.
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class AdvancedTextNormalizerPatch:
    """
    Patch class for AdvancedTextNormalizer to fix Story 4.1 issues:
    1. Fix 'str' object has no attribute 'groups' error
    2. Integrate with MCP Infrastructure Foundation
    3. Improve context classification consistency
    4. Fix capitalization issues in scriptural processing
    """
    
    def __init__(self, normalizer_instance):
        """Initialize patch for existing normalizer instance"""
        self.normalizer = normalizer_instance
        
        # Initialize missing attributes that cause errors
        self._initialize_missing_attributes()
        
        # Install MCP Infrastructure Foundation
        self._initialize_mcp_infrastructure()
        
        # Install patches
        self._install_patches()
        
        logger.info("AdvancedTextNormalizer patched for Story 4.1")
    
    def _initialize_missing_attributes(self):
        """Initialize missing attributes that cause runtime errors"""
        # Performance monitoring attributes
        if not hasattr(self.normalizer, 'target_processing_time_ms'):
            self.normalizer.target_processing_time_ms = 500.0
        
        # Confidence tracking
        if not hasattr(self.normalizer, 'confidence_tracking'):
            self.normalizer.confidence_tracking = {}
        
        # Confidence thresholds
        if not hasattr(self.normalizer, 'confidence_thresholds'):
            self.normalizer.confidence_thresholds = {
                'idiomatic': 0.85,
                'scriptural': 0.80,
                'temporal': 0.85,
                'mathematical': 0.75,
                'educational': 0.75,
                'ordinal': 0.75,
                'narrative': 0.80
            }
    
    def _initialize_mcp_infrastructure(self):
        """Initialize MCP Infrastructure Foundation"""
        try:
            from utils.mcp_infrastructure_foundation import create_mcp_infrastructure
            
            config = {
                'enable_mcp_processing': getattr(self.normalizer.config, 'enable_mcp_processing', True),
                'enable_fallback': getattr(self.normalizer.config, 'enable_fallback', True),
                'target_processing_time_ms': 500.0,
                'confidence_thresholds': self.normalizer.confidence_thresholds
            }
            
            self.normalizer._mcp_infrastructure = create_mcp_infrastructure(config)
            self.normalizer._initialize_mcp_client = lambda: None  # Already initialized
            
            logger.info("MCP Infrastructure Foundation integrated successfully")
            
        except Exception as e:
            logger.warning(f"MCP Infrastructure initialization failed: {e}")
            self.normalizer._mcp_infrastructure = None
            self.normalizer._initialize_mcp_client = lambda: logger.warning("MCP client unavailable")
    
    def _install_patches(self):
        """Install method patches to fix critical issues"""
        # Patch the math expression converter to handle string inputs
        self.normalizer._convert_math_expression = self._fixed_convert_math_expression
        self.normalizer._convert_math_equality = self._fixed_convert_math_equality
        
        # Patch scriptural processing to fix capitalization
        self.normalizer._convert_scriptural_numbers = self._fixed_convert_scriptural_numbers
        
        # Patch context classification for consistency
        self.normalizer._classify_number_context_enhanced = self._fixed_classify_number_context_enhanced
        
        # Patch main processing method
        original_convert = self.normalizer.convert_numbers_with_context
        self.normalizer.convert_numbers_with_context = self._enhanced_convert_numbers_with_context
        self.normalizer._original_convert_numbers_with_context = original_convert
        
        # Store reference to patch object for debugging
        self.normalizer._patch = self
    
    def _fixed_convert_math_expression(self, match_or_text):
        """Fixed math expression converter that handles both match objects and strings"""
        try:
            if hasattr(match_or_text, 'groups'):
                # It's a match object
                groups = match_or_text.groups()
                if len(groups) >= 3:
                    num1, operator, num2 = groups[0], groups[1], groups[2]
                else:
                    return match_or_text.group(0)
            else:
                # It's a string - parse manually
                import re
                pattern = r'\b(\w+)\s+(plus|minus|times|divided\s+by)\s+(\w+)\b'
                match = re.search(pattern, str(match_or_text), re.IGNORECASE)
                if match:
                    try:
                        groups = match.groups()
                        num1, operator, num2 = groups[0], groups[1], groups[2]
                    except AttributeError:
                        # Not a match object, return original
                        return str(match)
                else:
                    return str(match_or_text)
            
            # Convert to digits
            digit1 = self.normalizer._word_to_digit(num1)
            digit2 = self.normalizer._word_to_digit(num2)
            
            return f"{digit1} {operator} {digit2}"
            
        except Exception as e:
            logger.error(f"Math expression conversion error: {e}")
            return str(match_or_text) if hasattr(match_or_text, '__str__') else match_or_text.group(0)
    
    def _fixed_convert_math_equality(self, match_or_text):
        """Fixed math equality converter that handles both match objects and strings"""
        try:
            if hasattr(match_or_text, 'groups'):
                # It's a match object
                groups = match_or_text.groups()
                if len(groups) >= 3:
                    num1, operator, num2 = groups[0], groups[1], groups[2]
                else:
                    return match_or_text.group(0)
            else:
                # It's a string - parse manually
                import re
                pattern = r'\b(\w+)\s+(equals?|is)\s+(\w+)\b'
                match = re.search(pattern, str(match_or_text), re.IGNORECASE)
                if match:
                    try:
                        groups = match.groups()
                        num1, operator, num2 = groups[0], groups[1], groups[2]
                    except AttributeError:
                        # Not a match object, return original
                        return str(match)
                else:
                    return str(match_or_text)
            
            # Convert to digits
            digit1 = self.normalizer._word_to_digit(num1)
            digit2 = self.normalizer._word_to_digit(num2)
            
            return f"{digit1} {operator} {digit2}"
            
        except Exception as e:
            logger.error(f"Math equality conversion error: {e}")
            return str(match_or_text) if hasattr(match_or_text, '__str__') else match_or_text.group(0)
    
    def _fixed_convert_scriptural_numbers(self, text: str) -> str:
        """Fixed scriptural number conversion with proper capitalization"""
        try:
            # Handle case-sensitive scriptural patterns
            scriptural_patterns = [
                # Pattern: "chapter two verse twenty five" -> "Chapter 2 verse 25"
                (r'\b(chapter)\s+(\w+)(\s+verse\s+)(\w+(?:\s+\w+)?)\b', self._convert_scriptural_chapter_verse),
                # Pattern: "Bhagavad Gita chapter three" -> "Bhagavad Gita chapter 3"  
                (r'\b(Bhagavad\s+Gita\s+chapter)\s+(\w+)\b', self._convert_scriptural_reference),
                # Pattern: "verse three" -> "verse 3"
                (r'\b(verse)\s+(\w+(?:\s+\w+)?)\b', self._convert_verse_only)
            ]
            
            result = text
            for pattern, converter in scriptural_patterns:
                result = re.sub(pattern, converter, result, flags=re.IGNORECASE)
            
            return result
            
        except Exception as e:
            logger.error(f"Scriptural conversion error: {e}")
            return text
    
    def _convert_scriptural_chapter_verse(self, match):
        """Convert chapter and verse references with proper capitalization"""
        try:
            try:
                groups = match.groups()
            except AttributeError:
                # Not a match object, return original
                return str(match)
            if len(groups) >= 4:
                chapter_word = groups[0]  # "chapter"
                chapter_number = groups[1]  # "two"
                verse_connector = groups[2]  # " verse "
                verse_number = groups[3]  # "twenty five"
                
                # Convert numbers to digits
                chapter_digit = self.normalizer._word_to_digit(chapter_number)
                verse_digit = self.normalizer._word_to_digit(verse_number)
                
                # Always capitalize for proper scriptural formatting 
                return f"Chapter {chapter_digit}{verse_connector}{verse_digit}"
            
            return match.group(0)
            
        except Exception as e:
            logger.error(f"Chapter verse conversion error: {e}")
            return match.group(0)
    
    def _convert_scriptural_reference(self, match):
        """Convert scriptural references like 'Bhagavad Gita chapter three'"""
        try:
            try:
                groups = match.groups()
            except AttributeError:
                # Not a match object, return original
                return str(match)
            if len(groups) >= 2:
                reference_part = groups[0]  # "Bhagavad Gita chapter"
                number_part = groups[1]     # "three"
                
                number_digit = self.normalizer._word_to_digit(number_part)
                return f"{reference_part} {number_digit}"
            
            return match.group(0)
            
        except Exception as e:
            logger.error(f"Scriptural reference conversion error: {e}")
            return match.group(0)
    
    def _convert_verse_only(self, match):
        """Convert standalone verse references"""
        try:
            try:
                groups = match.groups()
            except AttributeError:
                # Not a match object, return original
                return str(match)
            if len(groups) >= 2:
                verse_word = groups[0]  # "verse"
                verse_number = groups[1]  # "three"
                
                verse_digit = self.normalizer._word_to_digit(verse_number)
                return f"{verse_word} {verse_digit}"
            
            return match.group(0)
            
        except Exception as e:
            logger.error(f"Verse conversion error: {e}")
            return match.group(0)
    
    def _fixed_classify_number_context_enhanced(self, text: str) -> Tuple[Any, float, List[Tuple[str, Any]]]:
        """Fixed context classification with improved consistency"""
        try:
            from utils.advanced_text_normalizer import NumberContextType
            
            # Enhanced classification patterns with higher precision
            classification_patterns = [
                # IDIOMATIC - Highest priority for quality preservation
                (r'\bone\s+by\s+one\b', NumberContextType.IDIOMATIC, 0.95),
                (r'\btwo\s+by\s+two\b', NumberContextType.IDIOMATIC, 0.95),
                (r'\bthree\s+by\s+three\b', NumberContextType.IDIOMATIC, 0.90),
                (r'\bstep\s+by\s+step\b', NumberContextType.IDIOMATIC, 0.90),
                (r'\bday\s+by\s+day\b', NumberContextType.IDIOMATIC, 0.90),
                (r'\bhand\s+in\s+hand\b', NumberContextType.IDIOMATIC, 0.90),
                (r'\bside\s+by\s+side\b', NumberContextType.IDIOMATIC, 0.90),
                
                # SCRIPTURAL - Religious/spiritual texts
                (r'\bchapter\s+\w+\s+verse\s+\w+', NumberContextType.SCRIPTURAL, 0.90),
                (r'\bbhagavad\s+gita\s+chapter\s+\w+', NumberContextType.SCRIPTURAL, 0.90),
                (r'\byoga\s+sutras?\s+chapter\s+\w+', NumberContextType.SCRIPTURAL, 0.85),
                (r'\bramayana\s+book\s+\w+', NumberContextType.SCRIPTURAL, 0.85),
                
                # TEMPORAL - Years and time references
                (r'\byear\s+two\s+thousand\s+\w+', NumberContextType.TEMPORAL, 0.95),
                (r'\bin\s+two\s+thousand\s+\w+', NumberContextType.TEMPORAL, 0.90),
                (r'\btwo\s+thousand\s+\w+', NumberContextType.TEMPORAL, 0.85),
                
                # MATHEMATICAL - Math expressions
                (r'\b\w+\s+(plus|minus|times|divided\s+by)\s+\w+', NumberContextType.MATHEMATICAL, 0.85),
                (r'\b\w+\s+(equals?|is)\s+\w+', NumberContextType.MATHEMATICAL, 0.80),
                (r'\bsum\s+of\s+\w+\s+and\s+\w+', NumberContextType.MATHEMATICAL, 0.80),
                
                # EDUCATIONAL - Learning contexts
                (r'\blesson\s+\w+', NumberContextType.EDUCATIONAL, 0.80),
                (r'\bpage\s+\w+', NumberContextType.EDUCATIONAL, 0.80),
                (r'\bgrade\s+\w+', NumberContextType.EDUCATIONAL, 0.80),
                (r'\bquestion\s+\w+', NumberContextType.EDUCATIONAL, 0.75),
                
                # ORDINAL - Sequence/order (preserve these patterns)
                (r'\b(first|second|third)\s+time', NumberContextType.ORDINAL, 0.95),
                (r'\b(first|second|third)\s+attempt', NumberContextType.ORDINAL, 0.95),
                
                # NARRATIVE - Story elements
                (r'\bonce\s+upon\s+a\s+time', NumberContextType.NARRATIVE, 0.85),
                (r'\b(first|second|third)\s+act', NumberContextType.NARRATIVE, 0.75),
            ]
            
            # Find the best matching pattern
            best_match = None
            best_confidence = 0.0
            best_context = NumberContextType.UNKNOWN
            
            for pattern, context_type, confidence in classification_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_context = context_type
                        best_match = pattern
            
            # Create segments (simplified for now)
            segments = [(text, best_context)]
            
            return best_context, best_confidence, segments
            
        except Exception as e:
            logger.error(f"Context classification error: {e}")
            return NumberContextType.UNKNOWN, 0.5, [(text, NumberContextType.UNKNOWN)]
    
    def _enhanced_convert_numbers_with_context(self, text: str) -> str:
        """Enhanced conversion with MCP infrastructure integration"""
        try:
            import time
            start_time = time.time()
            
            # Enhanced conversion processing for Story 4.1
            
            # Use MCP Infrastructure if available
            if hasattr(self.normalizer, '_mcp_infrastructure') and self.normalizer._mcp_infrastructure:
                try:
                    from utils.mcp_infrastructure_foundation import process_text_sync
                    
                    # Get context classification first
                    context_type, confidence, segments = self.normalizer._classify_number_context_enhanced(text)
                    
                    # Process with MCP Infrastructure
                    result = process_text_sync(
                        self.normalizer._mcp_infrastructure, 
                        text, 
                        context_type.value if hasattr(context_type, 'value') else str(context_type)
                    )
                    
                    if result['quality_passed'] and not result['errors']:
                        # Additional quality validation for Story 4.1 issues
                        mcp_text = result['processed_text']
                        if self._validate_story_4_1_quality(text, mcp_text, context_type):
                            return mcp_text
                        else:
                            logger.debug(f"MCP result failed Story 4.1 quality validation, using enhanced fallback")
                    else:
                        logger.debug(f"MCP processing failed: {result['errors']}")
                        
                except Exception as mcp_error:
                    logger.warning(f"MCP infrastructure error: {mcp_error}")
            
            # Enhanced fallback processing - always use our improved logic
            context_type, confidence, segments = self.normalizer._classify_number_context_enhanced(text)
            
            # Check for critical patterns first (highest priority)
            if self._has_critical_patterns(text):
                logger.debug(f"Preserving critical pattern in: {text}")
                return text
                
            # Apply context-specific processing with our fixes
            elif hasattr(context_type, 'value') and context_type.value == 'scriptural':
                # Apply scriptural processing with proper capitalization
                logger.debug(f"Applying scriptural processing to: {text}")
                return self.normalizer._convert_scriptural_numbers(text)
                
            elif hasattr(context_type, 'value') and context_type.value == 'mathematical':
                # Apply mathematical processing
                logger.debug(f"Applying mathematical processing to: {text}")
                return self._apply_mathematical_processing(text)
                
            elif hasattr(context_type, 'value') and context_type.value == 'ordinal':
                # Preserve ordinal patterns like "first time"
                logger.debug(f"Preserving ordinal pattern in: {text}")
                return text
                
            elif hasattr(context_type, 'value') and context_type.value == 'temporal':
                # Apply temporal processing with compound number fixes
                logger.debug(f"Applying temporal processing to: {text}")
                return self._apply_temporal_processing(text)
                
            else:
                # Use original number conversion for other contexts
                logger.debug(f"Using basic number conversion for: {text}")
                return self.normalizer.convert_numbers(text)
                    
        except Exception as e:
            logger.error(f"Enhanced conversion error: {e}")
            return text  # Safe fallback
    
    def _validate_story_4_1_quality(self, original_text: str, processed_text: str, context_type) -> bool:
        """Validate that MCP processing meets Story 4.1 quality standards"""
        try:
            # Quality check 1: Critical patterns must be preserved
            if self._has_critical_patterns(original_text):
                if original_text != processed_text:
                    logger.debug(f"Quality fail: Critical pattern not preserved: '{original_text}' -> '{processed_text}'")
                    return False
            
            # Quality check 2: Scriptural processing must include proper capitalization
            if hasattr(context_type, 'value') and context_type.value == 'scriptural':
                # Check for specific scriptural patterns that should be capitalized
                import re
                if re.search(r'\bchapter\s+\d+\s+verse\s+\d+', processed_text, re.IGNORECASE):
                    # Should start with capital "Chapter"
                    if not re.search(r'\bChapter\s+\d+\s+verse\s+\d+', processed_text):
                        logger.debug(f"Quality fail: Scriptural capitalization missing: '{processed_text}'")
                        return False
            
            # Quality check 3: Ordinal patterns in specific contexts should be preserved  
            if hasattr(context_type, 'value') and context_type.value == 'ordinal':
                # Check for ordinal patterns that should be preserved
                import re
                ordinal_preserve_patterns = [
                    r'\b(first|second|third)\s+time\b',
                    r'\b(first|second|third)\s+attempt\b'
                ]
                for pattern in ordinal_preserve_patterns:
                    if re.search(pattern, original_text, re.IGNORECASE):
                        # Should be preserved exactly
                        if not re.search(pattern, processed_text, re.IGNORECASE):
                            logger.debug(f"Quality fail: Ordinal pattern not preserved: '{original_text}' -> '{processed_text}'")
                            return False
            
            # Quality check 4: Temporal processing must properly handle compound numbers
            if hasattr(context_type, 'value') and context_type.value == 'temporal':
                import re
                # Check for temporal patterns that should be converted correctly
                temporal_quality_patterns = [
                    # "Year two thousand five" should become "Year 2005", NOT "Year 2000 5"
                    (r'\byear\s+two\s+thousand\s+(one|two|three|four|five|six|seven|eight|nine)\b', 
                     r'\bYear\s+200[1-9]\b'),
                    # General compound years should be properly converted  
                    (r'\btwo\s+thousand\s+(one|two|three|four|five|six|seven|eight|nine)\b',
                     r'\b200[1-9]\b')
                ]
                
                for original_pattern, expected_pattern in temporal_quality_patterns:
                    if re.search(original_pattern, original_text, re.IGNORECASE):
                        if not re.search(expected_pattern, processed_text):
                            logger.debug(f"Quality fail: Temporal compound number not properly converted: '{original_text}' -> '{processed_text}'")
                            return False
                        
                        # Additional check: ensure no broken patterns like "2000 5"
                        if re.search(r'\b2000\s+[1-9]\b', processed_text):
                            logger.debug(f"Quality fail: Temporal conversion created broken pattern: '{processed_text}'")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Quality validation error: {e}")
            return False  # Fail safe
    
    def _has_critical_patterns(self, text: str) -> bool:
        """Check for critical patterns that must be preserved"""
        critical_patterns = [
            r'\bone\s+by\s+one\b',
            r'\btwo\s+by\s+two\b',
            r'\bstep\s+by\s+step\b',
            r'\bday\s+by\s+day\b',
            # Ordinal patterns that should be preserved
            r'\b(first|second|third)\s+time\b',
            r'\b(first|second|third)\s+attempt\b'
        ]
        
        for pattern in critical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _apply_mathematical_processing(self, text: str) -> str:
        """Apply mathematical expression processing"""
        try:
            # Handle mathematical expressions like "two plus two equals four"
            math_patterns = [
                (r'\b(\w+)\s+(plus|minus|times|divided\s+by)\s+(\w+)\s+(equals?|is)\s+(\w+)\b', self._convert_full_math_expression),
                (r'\b(\w+)\s+(plus|minus|times|divided\s+by)\s+(\w+)\b', self._convert_math_operation),
                (r'\b(\w+)\s+(equals?|is)\s+(\w+)\b', self._convert_math_equality)
            ]
            
            result = text
            for pattern, converter in math_patterns:
                result = re.sub(pattern, converter, result, flags=re.IGNORECASE)
            
            return result
            
        except Exception as e:
            logger.error(f"Mathematical processing error: {e}")
            return text
    
    def _convert_full_math_expression(self, match):
        """Convert full math expressions like 'two plus two equals four'"""
        try:
            try:
                groups = match.groups()
            except AttributeError:
                # Not a match object, return original
                return str(match)
            if len(groups) >= 5:
                num1, operator, num2, equals_word, result_num = groups[0], groups[1], groups[2], groups[3], groups[4]
                
                digit1 = self.normalizer._word_to_digit(num1)
                digit2 = self.normalizer._word_to_digit(num2)
                result_digit = self.normalizer._word_to_digit(result_num)
                
                return f"{digit1} {operator} {digit2} {equals_word} {result_digit}"
            
            return match.group(0)
        except Exception as e:
            logger.error(f"Full math expression conversion error: {e}")
            return match.group(0)
    
    def _convert_math_operation(self, match):
        """Convert math operations like 'two plus two'"""
        try:
            try:
                groups = match.groups()
            except AttributeError:
                # Not a match object, return original
                return str(match)
            if len(groups) >= 3:
                num1, operator, num2 = groups[0], groups[1], groups[2]
                
                digit1 = self.normalizer._word_to_digit(num1)
                digit2 = self.normalizer._word_to_digit(num2)
                
                return f"{digit1} {operator} {digit2}"
            
            return match.group(0)
        except Exception as e:
            logger.error(f"Math operation conversion error: {e}")
            return match.group(0)

    
    def _apply_temporal_processing(self, text: str) -> str:
        """Apply temporal processing with enhanced compound number handling"""
        try:
            # Handle specific temporal patterns with compound numbers
            temporal_patterns = [
                # Handle "Year two thousand five" -> "Year 2005" (not "Year 2000 5")
                (r'\b(year)\s+(two\s+thousand\s+\w+)\b', self._convert_temporal_year),
                # Handle standalone years like "two thousand five" -> "2005"
                (r'\b(two\s+thousand\s+(one|two|three|four|five|six|seven|eight|nine))\b', self._convert_compound_year),
                # Handle other temporal patterns
                (r'\b(in\s+the\s+year)\s+(\w+(?:\s+\w+)*)\b', self._convert_temporal_in_year),
                # Handle decades like "nineteen ninety five" -> "1995"
                (r'\b(nineteen\s+(ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:\s+\w+)?)\b', self._convert_decades)
            ]
            
            result = text
            for pattern, converter in temporal_patterns:
                result = re.sub(pattern, converter, result, flags=re.IGNORECASE)
            
            # If no specific temporal patterns matched, use the base normalizer's _convert_temporal_numbers
            if result == text and hasattr(self.normalizer, '_convert_temporal_numbers'):
                result = self.normalizer._convert_temporal_numbers(text)
            
            return result
            
        except Exception as e:
            logger.error(f"Temporal processing error: {e}")
            return text
    
    def _convert_temporal_year(self, match):
        """Convert temporal year patterns like 'Year two thousand five'"""
        try:
            try:
                groups = match.groups()
            except AttributeError:
                # Not a match object, return original
                return str(match)
            if len(groups) >= 2:
                year_word = groups[0]  # "year"
                compound_number = groups[1]  # "two thousand five"
                
                # Handle compound numbers specifically for years
                year_digit = self._convert_compound_year_number(compound_number)
                
                # Capitalize "Year" for proper formatting
                return f"Year {year_digit}"
            
            return match.group(0)
            
        except Exception as e:
            logger.error(f"Temporal year conversion error: {e}")
            return match.group(0)
    
    def _convert_compound_year(self, match):
        """Convert compound year numbers like 'two thousand five'"""
        try:
            compound_number = match.group(0)
            return self._convert_compound_year_number(compound_number)
            
        except Exception as e:
            logger.error(f"Compound year conversion error: {e}")
            return match.group(0)
    
    def _convert_temporal_in_year(self, match):
        """Convert patterns like 'in the year two thousand five'"""
        try:
            try:
                groups = match.groups()
            except AttributeError:
                # Not a match object, return original
                return str(match)
            if len(groups) >= 2:
                in_year_phrase = groups[0]  # "in the year"
                year_words = groups[1]      # "two thousand five"
                
                year_digit = self._convert_compound_year_number(year_words)
                return f"{in_year_phrase} {year_digit}"
            
            return match.group(0)
            
        except Exception as e:
            logger.error(f"Temporal 'in year' conversion error: {e}")
            return match.group(0)
    
    def _convert_decades(self, match):
        """Convert decade patterns like 'nineteen ninety five'"""
        try:
            decade_text = match.group(0)
            
            # Handle decades like "nineteen ninety five" -> "1995"
            if "nineteen" in decade_text.lower():
                parts = decade_text.lower().split()
                if len(parts) >= 2:
                    if parts[1] in ["ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]:
                        decade_digit = self._decade_to_digit(parts[1])
                        if len(parts) >= 3:
                            # Handle "nineteen ninety five" -> "1995"
                            unit_digit = self.normalizer._word_to_digit(parts[2])
                            return f"19{decade_digit}{unit_digit}"
                        else:
                            # Handle "nineteen ninety" -> "1990"
                            return f"19{decade_digit}0"
            
            return match.group(0)
            
        except Exception as e:
            logger.error(f"Decade conversion error: {e}")
            return match.group(0)
    
    def _convert_compound_year_number(self, year_text: str) -> str:
        """Convert compound year numbers with proper handling of 'two thousand five' -> '2005'"""
        try:
            year_lower = year_text.lower().strip()
            
            # Handle "two thousand" + single digit
            if "two thousand" in year_lower:
                parts = year_lower.split()
                if len(parts) == 3 and parts[0] == "two" and parts[1] == "thousand":
                    # "two thousand five" -> "2005"
                    unit_digit = self.normalizer._word_to_digit(parts[2])
                    return f"200{unit_digit}"
                elif len(parts) == 2 and parts[0] == "two" and parts[1] == "thousand":
                    # "two thousand" -> "2000"
                    return "2000"
            
            # Handle other patterns like "one thousand nine hundred ninety five"
            if "thousand" in year_lower:
                parts = year_lower.split()
                thousand_idx = parts.index("thousand")
                
                if thousand_idx > 0:
                    thousands_part = " ".join(parts[:thousand_idx])
                    thousands_digit = self.normalizer._word_to_digit(thousands_part)
                    
                    if thousand_idx + 1 < len(parts):
                        remainder_part = " ".join(parts[thousand_idx + 1:])
                        remainder_digit = self.normalizer._word_to_digit(remainder_part)
                        return f"{thousands_digit}{remainder_digit:03d}"  # Pad with zeros
                    else:
                        return f"{thousands_digit}000"
            
            # Fallback to regular word-to-digit conversion
            return self.normalizer._word_to_digit(year_text)
            
        except Exception as e:
            logger.error(f"Compound year number conversion error: {e}")
            return year_text
    
    def _decade_to_digit(self, decade_word: str) -> str:
        """Convert decade words to digits"""
        decade_map = {
            'ten': '1', 'twenty': '2', 'thirty': '3', 'forty': '4', 'fifty': '5',
            'sixty': '6', 'seventy': '7', 'eighty': '8', 'ninety': '9'
        }
        return decade_map.get(decade_word.lower(), decade_word)

def apply_story_4_1_patch(normalizer_instance):
    """Apply Story 4.1 patches to AdvancedTextNormalizer instance"""
    patch = AdvancedTextNormalizerPatch(normalizer_instance)
    return normalizer_instance