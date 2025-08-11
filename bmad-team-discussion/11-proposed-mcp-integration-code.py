# Proposed MCP Integration Code - SOLUTION ARCHITECTURE
# This represents the enhanced system that would replace problematic logic

"""
SOLUTION OVERVIEW: Context-aware number processing using MCP linguistic intelligence

Key improvements:
1. Context classification before number conversion
2. Idiomatic expression preservation
3. Graceful fallback if MCP unavailable
4. Maintains all existing functionality
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class NumberContext(Enum):
    """Classification of number usage contexts."""
    IDIOMATIC = "idiomatic"        # "one by one", "step by step"
    MATHEMATICAL = "mathematical"   # "two thousand five", "three hundred"
    SCRIPTURAL = "scriptural"      # "chapter two verse twenty five"
    NARRATIVE = "narrative"        # "all the six children"
    TEMPORAL = "temporal"          # "year two thousand five"
    UNKNOWN = "unknown"

@dataclass
class ContextClassification:
    """Result of number context analysis."""
    original_phrase: str
    context_type: NumberContext
    confidence_score: float
    suggested_action: str  # 'preserve', 'convert', 'review'
    reasoning: str

class MCPClient:
    """Client for Model Context Protocol integration."""
    
    def __init__(self, server_configs: List[Dict]):
        """Initialize MCP clients for various NLP servers."""
        self.nlp_server = self._connect_server("nlp-server", server_configs)
        self.spacy_server = self._connect_server("spacy-server", server_configs)
        self.available = self._test_connectivity()
    
    def _connect_server(self, server_name: str, configs: List[Dict]):
        """Connect to specific MCP server."""
        # Implementation would use actual MCP client library
        pass
    
    def _test_connectivity(self) -> bool:
        """Test if MCP servers are available."""
        # Implementation would ping servers
        return True  # Placeholder
    
    def analyze_number_context(self, phrase: str, surrounding_text: str) -> ContextClassification:
        """
        Use MCP to analyze the context of a number phrase.
        
        This is the core intelligence that current system lacks.
        """
        if not self.available:
            return self._fallback_classification(phrase, surrounding_text)
        
        try:
            # Step 1: Tokenization and POS tagging via MCP
            tokens = self.spacy_server.tokenize(surrounding_text)
            
            # Step 2: Context analysis via MCP
            context_analysis = self.nlp_server.classify_context(
                phrase=phrase,
                tokens=tokens,
                task="number_normalization"
            )
            
            return ContextClassification(
                original_phrase=phrase,
                context_type=NumberContext(context_analysis.context),
                confidence_score=context_analysis.confidence,
                suggested_action=context_analysis.action,
                reasoning=context_analysis.explanation
            )
            
        except Exception as e:
            # Graceful degradation to rule-based
            return self._fallback_classification(phrase, surrounding_text)
    
    def _fallback_classification(self, phrase: str, surrounding_text: str) -> ContextClassification:
        """Fallback to enhanced rule-based classification."""
        # Enhanced rule-based patterns (better than current system)
        idiomatic_patterns = [
            r'\bone by one\b', r'\btwo by two\b', r'\bstep by step\b',
            r'\bone on one\b', r'\bday by day\b', r'\bone at a time\b'
        ]
        
        for pattern in idiomatic_patterns:
            if re.search(pattern, surrounding_text, re.IGNORECASE):
                return ContextClassification(
                    original_phrase=phrase,
                    context_type=NumberContext.IDIOMATIC,
                    confidence_score=0.9,
                    suggested_action="preserve",
                    reasoning="Matched idiomatic expression pattern"
                )
        
        # Default to mathematical context for conversion
        return ContextClassification(
            original_phrase=phrase,
            context_type=NumberContext.MATHEMATICAL,
            confidence_score=0.7,
            suggested_action="convert",
            reasoning="Default mathematical context"
        )

class AdvancedTextNormalizer:
    """
    Enhanced text normalizer with context-aware number processing.
    
    This replaces the problematic TextNormalizer.convert_numbers() method
    with intelligent context-aware processing.
    """
    
    def __init__(self, config: Dict):
        """Initialize with MCP integration configuration."""
        self.config = config
        self.mcp_client = MCPClient(config.get('mcp_servers', []))
        
        # Preserve existing number mappings for conversion when appropriate
        self._setup_number_mappings()
        
        # Context classification configuration
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.preserve_idiomatic = config.get('preserve_idiomatic', True)
    
    def _setup_number_mappings(self):
        """Same mappings as current system - used when conversion appropriate."""
        self.basic_numbers = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        
        self.compound_numbers = {
            "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
            "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90"
        }
    
    def normalize_numbers_with_context(self, text: str) -> str:
        """
        SOLUTION: Context-aware number processing that fixes quality issues.
        
        This method replaces the problematic convert_numbers() method.
        """
        # Find all potential number words in text
        number_words = self._find_number_words(text)
        
        # Classify each in context using MCP intelligence
        classifications = []
        for word_info in number_words:
            classification = self.mcp_client.analyze_number_context(
                phrase=word_info['phrase'],
                surrounding_text=text
            )
            classifications.append((word_info, classification))
        
        # Apply context-appropriate processing
        result_text = text
        for word_info, classification in classifications:
            result_text = self._apply_context_based_processing(
                result_text, word_info, classification
            )
        
        return result_text
    
    def _find_number_words(self, text: str) -> List[Dict]:
        """Find all number words and phrases in text."""
        number_words = []
        
        # Find basic number words
        for word in self.basic_numbers.keys():
            pattern = rf'\b{re.escape(word)}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                number_words.append({
                    'phrase': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'basic'
                })
        
        # Find compound expressions like "one by one"
        compound_patterns = [
            r'\b(one|two|three|four|five|six|seven|eight|nine) by \1\b',
            r'\b(step|day) by \1\b',
            r'\btwo thousand (one|two|three|four|five|six|seven|eight|nine)\b'
        ]
        
        for pattern in compound_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                number_words.append({
                    'phrase': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'compound'
                })
        
        return sorted(number_words, key=lambda x: x['start'])
    
    def _apply_context_based_processing(self, text: str, word_info: Dict, 
                                      classification: ContextClassification) -> str:
        """Apply appropriate processing based on context classification."""
        
        if classification.confidence_score < self.confidence_threshold:
            # Low confidence - preserve original
            return text
        
        if classification.context_type == NumberContext.IDIOMATIC:
            # PRESERVE idiomatic expressions - this fixes our quality issue!
            # "one by one" stays "one by one"
            return text
        
        elif classification.context_type == NumberContext.MATHEMATICAL:
            # CONVERT mathematical expressions
            # "two thousand five" → "2005"
            return self._convert_mathematical_number(text, word_info)
        
        elif classification.context_type == NumberContext.SCRIPTURAL:
            # PARTIAL CONVERT scriptural references
            # "chapter two verse twenty five" → "chapter 2 verse 25"
            return self._convert_scriptural_reference(text, word_info)
        
        elif classification.context_type == NumberContext.NARRATIVE:
            # CONTEXT-DEPENDENT narrative processing
            # "all the six children" → could be "all the 6 children" or preserved
            return self._handle_narrative_context(text, word_info, classification)
        
        else:
            # Unknown context - preserve to be safe
            return text
    
    def _convert_mathematical_number(self, text: str, word_info: Dict) -> str:
        """Convert mathematical number expressions."""
        phrase = word_info['phrase']
        
        # Use existing conversion logic for mathematical contexts
        if phrase.lower() in self.basic_numbers:
            replacement = self.basic_numbers[phrase.lower()]
            return text[:word_info['start']] + replacement + text[word_info['end']:]
        
        return text
    
    def _convert_scriptural_reference(self, text: str, word_info: Dict) -> str:
        """Convert scriptural references appropriately."""
        # Example: "chapter two verse twenty five" → "chapter 2 verse 25"
        # Implementation would handle verse/chapter references specifically
        return text
    
    def _handle_narrative_context(self, text: str, word_info: Dict, 
                                classification: ContextClassification) -> str:
        """Handle narrative contexts with style preferences."""
        # Could preserve "six children" vs convert to "6 children" based on config
        style_preference = self.config.get('narrative_number_style', 'preserve')
        
        if style_preference == 'preserve':
            return text  # Keep "six children"
        else:
            return self._convert_mathematical_number(text, word_info)

# Integration with existing SanskritPostProcessor
"""
Enhanced integration in src/post_processors/sanskrit_post_processor.py:

class SanskritPostProcessor:
    def __init__(self, config):
        # Existing initialization...
        
        # NEW: Initialize advanced normalizer if configured
        if config.get('enable_advanced_normalization', False):
            self.advanced_normalizer = AdvancedTextNormalizer(config)
        else:
            self.advanced_normalizer = None
    
    def _process_srt_segment(self, segment, file_metrics):
        # Existing preprocessing...
        
        # ENHANCED: Context-aware normalization
        if self.advanced_normalizer:
            # Use MCP-enhanced processing
            normalized_result = self.advanced_normalizer.normalize_with_context(text)
        else:
            # Fallback to current system
            normalized_result = self.text_normalizer.normalize_with_advanced_tracking(text)
        
        text = normalized_result.corrected_text
        
        # Existing NER and other processing continues unchanged...
"""

# Configuration example
"""
# config/advanced_normalization_config.yaml
mcp_integration:
  enable_advanced_normalization: true
  mcp_servers:
    - name: "nlp-server"
      endpoint: "mcp://localhost:8000"
      capabilities: ["context_analysis", "pos_tagging"]
    - name: "spacy-server"
      endpoint: "mcp://localhost:8001"
      capabilities: ["tokenization", "linguistic_analysis"]
  
  context_classification:
    confidence_threshold: 0.8
    preserve_idiomatic: true
    narrative_number_style: "preserve"  # or "convert"
  
  fallback_strategy:
    enable_enhanced_rules: true
    fallback_on_mcp_failure: true
"""