"""
Contextual Number Processor for spiritual and academic contexts.

This module provides specialized number processing capabilities for spiritual texts,
including scriptural references, dates, times, and contextual number conversion
with awareness of spiritual terminology.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum


class NumberContext(Enum):
    """Types of number contexts in spiritual texts."""
    SCRIPTURAL_REFERENCE = "scriptural_reference"  # Chapter 2 verse 25
    DATE = "date"  # January first, two thousand five
    TIME = "time"  # quarter past two
    ORDINAL = "ordinal"  # first, second, third
    CARDINAL = "cardinal"  # one, two, three
    YEAR = "year"  # nineteen ninety five
    AGE = "age"  # twenty years old
    COUNT = "count"  # three times


@dataclass
class NumberConversion:
    """Represents a number conversion with context."""
    original_text: str
    converted_text: str
    number_context: NumberContext
    confidence_score: float
    start_pos: int
    end_pos: int
    reasoning: str


@dataclass
class ConversionResult:
    """Result of contextual number processing."""
    original_text: str
    processed_text: str
    conversions: List[NumberConversion]
    total_conversions: int
    high_confidence_conversions: int
    processing_notes: List[str]


class ContextualNumberProcessor:
    """
    Processes numbers with awareness of spiritual and academic contexts.
    
    This processor handles various number formats commonly found in spiritual
    lectures and academic discussions, with special attention to scriptural
    references and religious terminology.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the contextual number processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Processing thresholds
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.7)
        self.preserve_uncertainty = self.config.get('preserve_uncertainty', True)
        
        # Initialize number mappings and patterns
        self._setup_basic_number_mappings()
        self._setup_spiritual_context_patterns()
        self._setup_date_time_patterns()
        self._setup_ordinal_mappings()
        self._setup_scriptural_reference_patterns()
    
    def process_numbers(self, text: str, context: str = "spiritual") -> ConversionResult:
        """
        Process numbers in text with contextual awareness.
        
        Args:
            text: Input text to process
            context: Context hint ("spiritual", "academic", "general")
            
        Returns:
            ConversionResult with all number conversions
        """
        if not text or not text.strip():
            return ConversionResult(
                original_text=text,
                processed_text=text,
                conversions=[],
                total_conversions=0,
                high_confidence_conversions=0,
                processing_notes=["Empty or whitespace-only text"]
            )
        
        conversions = []
        processing_notes = []
        
        # Detect different types of number patterns
        conversions.extend(self._process_scriptural_references(text))
        conversions.extend(self._process_date_expressions(text))
        conversions.extend(self._process_time_expressions(text))
        conversions.extend(self._process_ordinal_numbers(text))
        conversions.extend(self._process_cardinal_numbers(text))
        conversions.extend(self._process_year_expressions(text))
        
        # Sort conversions by position for consistent processing
        conversions.sort(key=lambda c: c.start_pos)
        
        # Resolve overlapping conversions
        conversions = self._resolve_overlapping_conversions(conversions)
        
        # Apply high-confidence conversions
        processed_text = self._apply_conversions(text, conversions)
        
        # Calculate statistics
        high_confidence_conversions = sum(
            1 for c in conversions if c.confidence_score >= self.min_confidence_threshold
        )
        
        if conversions:
            processing_notes.append(f"Detected {len(conversions)} number patterns")
            processing_notes.append(f"{high_confidence_conversions} high-confidence conversions")
        
        return ConversionResult(
            original_text=text,
            processed_text=processed_text,
            conversions=conversions,
            total_conversions=len(conversions),
            high_confidence_conversions=high_confidence_conversions,
            processing_notes=processing_notes
        )
    
    def _process_scriptural_references(self, text: str) -> List[NumberConversion]:
        """Process scriptural references like 'chapter two verse twenty five'."""
        conversions = []
        
        for pattern in self.scriptural_patterns:
            matches = list(re.finditer(pattern['regex'], text, re.IGNORECASE))
            
            for match in matches:
                conversion = self._convert_scriptural_reference(match, pattern)
                if conversion:
                    conversions.append(conversion)
        
        return conversions
    
    def _process_date_expressions(self, text: str) -> List[NumberConversion]:
        """Process date expressions like 'January first, two thousand five'."""
        conversions = []
        
        for pattern in self.date_patterns:
            matches = list(re.finditer(pattern['regex'], text, re.IGNORECASE))
            
            for match in matches:
                conversion = self._convert_date_expression(match, pattern)
                if conversion:
                    conversions.append(conversion)
        
        return conversions
    
    def _process_time_expressions(self, text: str) -> List[NumberConversion]:
        """Process time expressions like 'quarter past two'."""
        conversions = []
        
        for pattern in self.time_patterns:
            matches = list(re.finditer(pattern['regex'], text, re.IGNORECASE))
            
            for match in matches:
                conversion = self._convert_time_expression(match, pattern)
                if conversion:
                    conversions.append(conversion)
        
        return conversions
    
    def _process_ordinal_numbers(self, text: str) -> List[NumberConversion]:
        """Process ordinal numbers in spiritual contexts."""
        conversions = []
        
        # Enhanced ordinal processing for spiritual contexts
        spiritual_ordinal_pattern = r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty\s+first|twenty\s+second|twenty\s+third|twenty\s+fourth|twenty\s+fifth)\s+(chapter|verse|book|canto|section|part)\b'
        
        matches = list(re.finditer(spiritual_ordinal_pattern, text, re.IGNORECASE))
        
        for match in matches:
            ordinal_word = match.group(1).lower().replace(' ', '')
            context_word = match.group(2).lower()
            
            # Convert ordinal word to number
            ordinal_number = self._convert_ordinal_word_to_number(ordinal_word)
            
            if ordinal_number:
                converted_text = f"{ordinal_number} {context_word}"
                
                conversion = NumberConversion(
                    original_text=match.group(0),
                    converted_text=converted_text,
                    number_context=NumberContext.ORDINAL,
                    confidence_score=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    reasoning=f"Spiritual context ordinal: {context_word}"
                )
                conversions.append(conversion)
        
        return conversions
    
    def _process_cardinal_numbers(self, text: str) -> List[NumberConversion]:
        """Process cardinal numbers with spiritual context awareness."""
        conversions = []
        
        # Process compound numbers first (e.g., "twenty five")
        compound_pattern = r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\b'
        matches = list(re.finditer(compound_pattern, text, re.IGNORECASE))
        
        for match in matches:
            tens_word = match.group(1).lower()
            ones_word = match.group(2).lower()
            
            tens_digit = self.basic_numbers.get(tens_word, 0)
            ones_digit = self.basic_numbers.get(ones_word, 0)
            
            if tens_digit and ones_digit:
                total = int(tens_digit) + int(ones_digit)
                
                conversion = NumberConversion(
                    original_text=match.group(0),
                    converted_text=str(total),
                    number_context=NumberContext.CARDINAL,
                    confidence_score=0.95,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    reasoning="Compound cardinal number"
                )
                conversions.append(conversion)
        
        # Process basic numbers in spiritual contexts
        for word, digit in self.basic_numbers.items():
            # Look for the word with spiritual context indicators
            spiritual_context_pattern = rf'\\b{re.escape(word)}\\s+(chapter|verse|book|canto|time|year|age)\\b'
            matches = list(re.finditer(spiritual_context_pattern, text, re.IGNORECASE))
            
            for match in matches:
                context_word = match.group(1).lower()
                converted_text = f"{digit} {context_word}"
                
                conversion = NumberConversion(
                    original_text=match.group(0),
                    converted_text=converted_text,
                    number_context=NumberContext.CARDINAL,
                    confidence_score=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    reasoning=f"Cardinal number with spiritual context: {context_word}"
                )
                conversions.append(conversion)
        
        return conversions
    
    def _process_year_expressions(self, text: str) -> List[NumberConversion]:
        """Process year expressions like 'two thousand five'."""
        conversions = []
        
        for pattern in self.year_patterns:
            matches = list(re.finditer(pattern['regex'], text, re.IGNORECASE))
            
            for match in matches:
                conversion = self._convert_year_expression(match, pattern)
                if conversion:
                    conversions.append(conversion)
        
        return conversions
    
    def _convert_scriptural_reference(self, match, pattern: Dict) -> Optional[NumberConversion]:
        """Convert a scriptural reference match to numbers."""
        try:
            reference_type = pattern['type']
            
            if reference_type == 'chapter_verse':
                # Pattern: "chapter two verse twenty five"
                try:
                    groups = match.groups()
                except AttributeError:
                    return str(match)
                if len(groups) >= 2:
                    chapter_word = groups[0].lower()
                    verse_word = groups[1].lower()
                    
                    chapter_num = self._word_to_number(chapter_word)
                    verse_num = self._word_to_number(verse_word)
                    
                    if chapter_num and verse_num:
                        converted_text = f"chapter {chapter_num} verse {verse_num}"
                        
                        return NumberConversion(
                            original_text=match.group(0),
                            converted_text=converted_text,
                            number_context=NumberContext.SCRIPTURAL_REFERENCE,
                            confidence_score=0.95,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            reasoning="Scriptural chapter and verse reference"
                        )
            
            elif reference_type == 'verse_only':
                # Pattern: "verse twenty five"
                try:
                    groups = match.groups()
                except AttributeError:
                    return str(match)
                if groups:
                    verse_word = groups[0].lower()
                    verse_num = self._word_to_number(verse_word)
                    
                    if verse_num:
                        converted_text = f"verse {verse_num}"
                        
                        return NumberConversion(
                            original_text=match.group(0),
                            converted_text=converted_text,
                            number_context=NumberContext.SCRIPTURAL_REFERENCE,
                            confidence_score=0.9,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            reasoning="Scriptural verse reference"
                        )
        
        except Exception as e:
            self.logger.warning(f"Error converting scriptural reference: {e}")
        
        return None
    
    def _convert_date_expression(self, match, pattern: Dict) -> Optional[NumberConversion]:
        """Convert a date expression match to standard format."""
        try:
            date_type = pattern['type']
            
            if date_type == 'month_day_year':
                # Pattern: "January first, two thousand five"
                try:
                    groups = match.groups()
                except AttributeError:
                    return str(match)
                if len(groups) >= 3:
                    month = groups[0]
                    day_word = groups[1].lower()
                    year_words = groups[2].lower()
                    
                    day_num = self._convert_ordinal_word_to_number(day_word)
                    year_num = self._convert_year_words_to_number(year_words)
                    
                    if day_num and year_num:
                        converted_text = f"{month} {day_num}, {year_num}"
                        
                        return NumberConversion(
                            original_text=match.group(0),
                            converted_text=converted_text,
                            number_context=NumberContext.DATE,
                            confidence_score=0.85,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            reasoning="Date expression with month, day, and year"
                        )
        
        except Exception as e:
            self.logger.warning(f"Error converting date expression: {e}")
        
        return None
    
    def _convert_time_expression(self, match, pattern: Dict) -> Optional[NumberConversion]:
        """Convert a time expression match to standard format."""
        try:
            time_type = pattern['type']
            
            if time_type == 'quarter_past':
                # Pattern: "quarter past two"
                try:
                    groups = match.groups()
                except AttributeError:
                    return str(match)
                if groups:
                    hour_word = groups[0].lower()
                    hour_num = self.basic_numbers.get(hour_word)
                    
                    if hour_num:
                        converted_text = f"{hour_num}:15"
                        
                        return NumberConversion(
                            original_text=match.group(0),
                            converted_text=converted_text,
                            number_context=NumberContext.TIME,
                            confidence_score=0.9,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            reasoning="Time expression: quarter past"
                        )
            
            elif time_type == 'half_past':
                # Pattern: "half past three"
                try:
                    groups = match.groups()
                except AttributeError:
                    return str(match)
                if groups:
                    hour_word = groups[0].lower()
                    hour_num = self.basic_numbers.get(hour_word)
                    
                    if hour_num:
                        converted_text = f"{hour_num}:30"
                        
                        return NumberConversion(
                            original_text=match.group(0),
                            converted_text=converted_text,
                            number_context=NumberContext.TIME,
                            confidence_score=0.9,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            reasoning="Time expression: half past"
                        )
        
        except Exception as e:
            self.logger.warning(f"Error converting time expression: {e}")
        
        return None
    
    def _convert_year_expression(self, match, pattern: Dict) -> Optional[NumberConversion]:
        """Convert a year expression match to digits."""
        try:
            year_type = pattern['type']
            
            if year_type == 'two_thousand':
                # Pattern: "two thousand five"
                try:
                    groups = match.groups()
                except AttributeError:
                    return str(match)
                if groups:
                    remainder_word = groups[0].lower() if groups[0] else ''
                    
                    if remainder_word:
                        remainder_num = self.basic_numbers.get(remainder_word, 0)
                        year_num = 2000 + int(remainder_num)
                    else:
                        year_num = 2000
                    
                    return NumberConversion(
                        original_text=match.group(0),
                        converted_text=str(year_num),
                        number_context=NumberContext.YEAR,
                        confidence_score=0.9,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        reasoning="Year expression: two thousand format"
                    )
            
            elif year_type == 'nineteen_hundreds':
                # Pattern: "nineteen ninety five"
                try:
                    groups = match.groups()
                except AttributeError:
                    return str(match)
                if len(groups) >= 2:
                    tens_word = groups[0].lower()
                    ones_word = groups[1].lower()
                    
                    tens_num = self.basic_numbers.get(tens_word, 0)
                    ones_num = self.basic_numbers.get(ones_word, 0)
                    
                    if tens_num and ones_num:
                        year_num = 1900 + int(tens_num) + int(ones_num)
                        
                        return NumberConversion(
                            original_text=match.group(0),
                            converted_text=str(year_num),
                            number_context=NumberContext.YEAR,
                            confidence_score=0.85,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            reasoning="Year expression: nineteen hundreds format"
                        )
        
        except Exception as e:
            self.logger.warning(f"Error converting year expression: {e}")
        
        return None
    
    def _word_to_number(self, word: str) -> Optional[str]:
        """Convert a word to its numeric representation."""
        word = word.lower().strip()
        
        # Check basic numbers first
        if word in self.basic_numbers:
            return self.basic_numbers[word]
        
        # Check compound numbers
        compound_match = re.match(r'(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*(one|two|three|four|five|six|seven|eight|nine)?', word)
        if compound_match:
            tens_word = compound_match.group(1)
            ones_word = compound_match.group(2) or '0'
            
            tens_digit = int(self.basic_numbers.get(tens_word, 0))
            ones_digit = int(self.basic_numbers.get(ones_word, 0))
            
            return str(tens_digit + ones_digit)
        
        return None
    
    def _convert_ordinal_word_to_number(self, word: str) -> Optional[str]:
        """Convert ordinal word to number (e.g., 'first' -> '1st')."""
        word = word.lower().strip()
        
        if word in self.ordinal_mappings:
            return self.ordinal_mappings[word]
        
        # Handle compound ordinals like "twenty first"
        compound_match = re.match(r'(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)', word)
        if compound_match:
            tens_word = compound_match.group(1)
            ordinal_word = compound_match.group(2)
            
            tens_digit = int(self.basic_numbers.get(tens_word, 0))
            ones_ordinal = self.ordinal_mappings.get(ordinal_word, '1st')
            ones_digit = int(re.match(r'(\d+)', ones_ordinal).group(1))
            
            total = tens_digit + ones_digit
            
            # Generate proper ordinal suffix
            if 10 <= total % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(total % 10, 'th')
            
            return f"{total}{suffix}"
        
        return None
    
    def _convert_year_words_to_number(self, year_words: str) -> Optional[str]:
        """Convert year words to numeric year."""
        year_words = year_words.lower().strip()
        
        # Handle "two thousand X" patterns
        two_thousand_match = re.match(r'two\s+thousand\s*(?:and\s+)?(\w+)?', year_words)
        if two_thousand_match:
            remainder = two_thousand_match.group(1)
            if remainder:
                remainder_num = self.basic_numbers.get(remainder, 0)
                return str(2000 + int(remainder_num))
            else:
                return "2000"
        
        # Handle "nineteen X Y" patterns
        nineteen_match = re.match(r'nineteen\s+(\w+)\s*(\w+)?', year_words)
        if nineteen_match:
            tens_word = nineteen_match.group(1)
            ones_word = nineteen_match.group(2) or 'zero'
            
            tens_num = self.basic_numbers.get(tens_word, 0)
            ones_num = self.basic_numbers.get(ones_word, 0)
            
            return str(1900 + int(tens_num) + int(ones_num))
        
        return None
    
    def _setup_basic_number_mappings(self):
        """Setup basic number word to digit mappings."""
        self.basic_numbers = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90'
        }
    
    def _setup_spiritual_context_patterns(self):
        """Setup patterns specific to spiritual contexts."""
        self.spiritual_keywords = {
            'chapter', 'verse', 'book', 'canto', 'section', 'part',
            'sloka', 'mantra', 'sutra', 'upanishad', 'gita'
        }
    
    def _setup_date_time_patterns(self):
        """Setup date and time pattern definitions."""
        self.date_patterns = [
            {
                'regex': r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty\s+first|twenty\s+second|twenty\s+third|twenty\s+fourth|twenty\s+fifth|twenty\s+sixth|twenty\s+seventh|twenty\s+eighth|twenty\s+ninth|thirtieth|thirty\s+first),?\s+((?:two\s+thousand\s*(?:and\s+)?\w*)|(?:nineteen\s+\w+\s*\w*))\b',
                'type': 'month_day_year'
            }
        ]
        
        self.time_patterns = [
            {
                'regex': r'\bquarter\s+past\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b',
                'type': 'quarter_past'
            },
            {
                'regex': r'\bhalf\s+past\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b',
                'type': 'half_past'
            }
        ]
    
    def _setup_ordinal_mappings(self):
        """Setup ordinal word mappings."""
        self.ordinal_mappings = {
            'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
            'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
            'ninth': '9th', 'tenth': '10th', 'eleventh': '11th', 'twelfth': '12th',
            'thirteenth': '13th', 'fourteenth': '14th', 'fifteenth': '15th',
            'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
            'nineteenth': '19th', 'twentieth': '20th', 'twenty-first': '21st',
            'twenty-second': '22nd', 'twenty-third': '23rd', 'twenty-fourth': '24th',
            'twenty-fifth': '25th', 'thirtieth': '30th'
        }
    
    def _setup_scriptural_reference_patterns(self):
        """Setup scriptural reference pattern definitions."""
        self.scriptural_patterns = [
            {
                'regex': r'\bchapter\s+(twenty\s+one|twenty\s+two|twenty\s+three|twenty\s+four|twenty\s+five|twenty\s+six|twenty\s+seven|twenty\s+eight|twenty\s+nine|thirty\s+one|thirty\s+two|thirty\s+three|thirty\s+four|thirty\s+five|thirty\s+six|thirty\s+seven|thirty\s+eight|thirty\s+nine|forty\s+one|forty\s+two|forty\s+three|forty\s+four|forty\s+five|forty\s+six|forty\s+seven|forty\s+eight|forty\s+nine|fifty\s+one|fifty\s+two|fifty\s+three|fifty\s+four|fifty\s+five|fifty\s+six|fifty\s+seven|fifty\s+eight|fifty\s+nine|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+verse\s+(twenty\s+one|twenty\s+two|twenty\s+three|twenty\s+four|twenty\s+five|twenty\s+six|twenty\s+seven|twenty\s+eight|twenty\s+nine|thirty\s+one|thirty\s+two|thirty\s+three|thirty\s+four|thirty\s+five|thirty\s+six|thirty\s+seven|thirty\s+eight|thirty\s+nine|forty\s+one|forty\s+two|forty\s+three|forty\s+four|forty\s+five|forty\s+six|forty\s+seven|forty\s+eight|forty\s+nine|fifty\s+one|fifty\s+two|fifty\s+three|fifty\s+four|fifty\s+five|fifty\s+six|fifty\s+seven|fifty\s+eight|fifty\s+nine|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b',
                'type': 'chapter_verse'
            },
            {
                'regex': r'\bverse\s+(twenty\s+one|twenty\s+two|twenty\s+three|twenty\s+four|twenty\s+five|twenty\s+six|twenty\s+seven|twenty\s+eight|twenty\s+nine|thirty\s+one|thirty\s+two|thirty\s+three|thirty\s+four|thirty\s+five|thirty\s+six|thirty\s+seven|thirty\s+eight|thirty\s+nine|forty\s+one|forty\s+two|forty\s+three|forty\s+four|forty\s+five|forty\s+six|forty\s+seven|forty\s+eight|forty\s+nine|fifty\s+one|fifty\s+two|fifty\s+three|fifty\s+four|fifty\s+five|fifty\s+six|fifty\s+seven|fifty\s+eight|fifty\s+nine|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b',
                'type': 'verse_only'
            }
        ]
        
        self.year_patterns = [
            {
                'regex': r'\btwo\s+thousand\s*(?:and\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)?\b',
                'type': 'two_thousand'
            },
            {
                'regex': r'\bnineteen\s+(ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*(one|two|three|four|five|six|seven|eight|nine)?\b',
                'type': 'nineteen_hundreds'
            }
        ]
    
    def _apply_conversions(self, text: str, conversions: List[NumberConversion]) -> str:
        """Apply number conversions to text."""
        if not conversions:
            return text
        
        # Filter to high-confidence conversions only
        high_confidence_conversions = [
            c for c in conversions
            if c.confidence_score >= self.min_confidence_threshold
        ]
        
        # Sort by position (reverse order for safe string replacement)
        high_confidence_conversions.sort(key=lambda c: c.start_pos, reverse=True)
        
        processed_text = text
        
        for conversion in high_confidence_conversions:
            try:
                processed_text = (
                    processed_text[:conversion.start_pos] +
                    conversion.converted_text +
                    processed_text[conversion.end_pos:]
                )
                
                self.logger.debug(
                    f"Applied {conversion.number_context.value} conversion: "
                    f"'{conversion.original_text}' -> '{conversion.converted_text}'"
                )
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to apply number conversion: {e}. "
                    f"Conversion: {conversion.number_context.value} at {conversion.start_pos}-{conversion.end_pos}"
                )
        
        return processed_text
    
    def _resolve_overlapping_conversions(self, conversions: List[NumberConversion]) -> List[NumberConversion]:
        """Resolve overlapping conversions by keeping the highest confidence ones."""
        if not conversions:
            return conversions
        
        resolved_conversions = []
        sorted_conversions = sorted(conversions, key=lambda c: c.confidence_score, reverse=True)
        
        for conversion in sorted_conversions:
            # Check if this conversion overlaps with any already accepted conversion
            overlaps = False
            for accepted_conversion in resolved_conversions:
                if (conversion.start_pos < accepted_conversion.end_pos and 
                    conversion.end_pos > accepted_conversion.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                resolved_conversions.append(conversion)
        
        # Sort by position for consistent processing
        resolved_conversions.sort(key=lambda c: c.start_pos)
        
        return resolved_conversions