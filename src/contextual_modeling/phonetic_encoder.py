"""
Phonetic Encoding System for Sanskrit/Hindi Text

This module provides phonetic representation capabilities for improved matching
between ASR output and lexicon entries using various phonetic encoding algorithms
optimized for Sanskrit and Hindi pronunciation patterns.
"""

import re
import string
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

from utils.logger_config import get_logger


class PhoneticAlgorithm(Enum):
    """Available phonetic encoding algorithms."""
    SOUNDEX = "soundex"
    METAPHONE = "metaphone"
    DOUBLE_METAPHONE = "double_metaphone"
    SANSKRIT_PHONETIC = "sanskrit_phonetic"
    HYBRID = "hybrid"


@dataclass
class PhoneticConfig:
    """Configuration for phonetic encoding."""
    algorithm: PhoneticAlgorithm = PhoneticAlgorithm.SANSKRIT_PHONETIC
    max_code_length: int = 8
    enable_devanagari_mapping: bool = True
    enable_aspiration_normalization: bool = True
    enable_vowel_length_normalization: bool = True
    similarity_threshold: float = 0.8
    enable_compound_encoding: bool = True


@dataclass
class PhoneticMatch:
    """Phonetic matching result."""
    original_text: str
    target_text: str
    original_code: str
    target_code: str
    similarity_score: float
    algorithm_used: PhoneticAlgorithm
    confidence: float


class PhoneticEncoder:
    """
    Advanced phonetic encoding system for Sanskrit/Hindi text.
    
    Provides multiple phonetic encoding algorithms with special handling
    for Sanskrit phonetic patterns, aspiration, and vowel length.
    """

    def __init__(self, config: PhoneticConfig = None):
        """
        Initialize phonetic encoder.
        
        Args:
            config: Phonetic encoding configuration
        """
        self.config = config or PhoneticConfig()
        self.logger = get_logger(__name__)
        
        # Initialize phonetic mappings
        self._init_sanskrit_mappings()
        self._init_devanagari_mappings()
        self._init_aspiration_mappings()
        
        self.logger.info(f"PhoneticEncoder initialized with {self.config.algorithm.value}")

    def _init_sanskrit_mappings(self) -> None:
        """Initialize Sanskrit-specific phonetic mappings."""
        # Consonant clusters and their simplified forms
        self.consonant_clusters = {
            'ksh': 'ks', 'gny': 'gn', 'ngy': 'ng',
            'ttr': 'tr', 'ddr': 'dr', 'nnr': 'nr',
            'mbh': 'bh', 'ngh': 'gh', 'ndh': 'dh',
            'rth': 'th', 'rph': 'ph', 'rbh': 'bh'
        }
        
        # Retroflex to dental mappings for ASR confusion
        self.retroflex_mappings = {
            'ṭ': 't', 'ṭh': 'th', 'ḍ': 'd', 'ḍh': 'dh', 'ṇ': 'n',
            'ṛ': 'r', 'ṝ': 'r', 'ḷ': 'l', 'ḹ': 'l'
        }
        
        # Sibilant variations
        self.sibilant_mappings = {
            'ś': 's', 'ṣ': 's', 'kṣ': 'ks', 'jñ': 'gn'
        }
        
        # Common Sanskrit phonetic substitutions
        self.phonetic_substitutions = [
            ('v', 'w'), ('w', 'v'),
            ('ri', 'r'), ('ru', 'r'),
            ('ai', 'e'), ('au', 'o'),
            ('kh', 'k'), ('gh', 'g'),
            ('ch', 'c'), ('jh', 'j'),
            ('th', 't'), ('dh', 'd'),
            ('ph', 'p'), ('bh', 'b')
        ]

    def _init_devanagari_mappings(self) -> None:
        """Initialize Devanagari to Roman mappings."""
        self.devanagari_to_roman = {
            # Vowels
            'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii',
            'उ': 'u', 'ऊ': 'uu', 'ऋ': 'r', 'ॠ': 'rr',
            'ऌ': 'l', 'ॡ': 'll', 'ए': 'e', 'ऐ': 'ai',
            'ओ': 'o', 'औ': 'au',
            
            # Consonants
            'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
            'च': 'c', 'छ': 'ch', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
            'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
            'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
            'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
            'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
            'श': 's', 'ष': 's', 'स': 's', 'ह': 'h',
            
            # Matras (vowel signs)
            'ा': 'aa', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
            'ृ': 'r', 'ॄ': 'rr', 'ॢ': 'l', 'ॣ': 'll',
            'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
            
            # Special characters
            'ं': 'm', 'ः': 'h', '्': ''
        }

    def _init_aspiration_mappings(self) -> None:
        """Initialize aspiration normalization mappings."""
        self.aspiration_pairs = [
            ('kh', 'k'), ('gh', 'g'), ('ch', 'c'), ('jh', 'j'),
            ('th', 't'), ('dh', 'd'), ('ph', 'p'), ('bh', 'b'),
            ('ṭh', 'ṭ'), ('ḍh', 'ḍ')
        ]

    def encode_text(self, text: str) -> str:
        """
        Encode text using the configured phonetic algorithm.
        
        Args:
            text: Text to encode
            
        Returns:
            Phonetic code
        """
        if self.config.algorithm == PhoneticAlgorithm.SANSKRIT_PHONETIC:
            return self._sanskrit_phonetic_encode(text)
        elif self.config.algorithm == PhoneticAlgorithm.SOUNDEX:
            return self._soundex_encode(text)
        elif self.config.algorithm == PhoneticAlgorithm.HYBRID:
            return self._hybrid_encode(text)
        else:
            # Default to Sanskrit phonetic
            return self._sanskrit_phonetic_encode(text)

    def _sanskrit_phonetic_encode(self, text: str) -> str:
        """
        Sanskrit-optimized phonetic encoding.
        
        Args:
            text: Text to encode
            
        Returns:
            Sanskrit phonetic code
        """
        # Preprocessing
        text = self._preprocess_text(text)
        
        # Convert Devanagari if present
        if self.config.enable_devanagari_mapping:
            text = self._convert_devanagari(text)
        
        # Normalize aspiration
        if self.config.enable_aspiration_normalization:
            text = self._normalize_aspiration(text)
        
        # Normalize vowel lengths
        if self.config.enable_vowel_length_normalization:
            text = self._normalize_vowel_lengths(text)
        
        # Apply retroflex mappings
        text = self._apply_retroflex_mappings(text)
        
        # Apply sibilant mappings
        text = self._apply_sibilant_mappings(text)
        
        # Handle consonant clusters
        text = self._handle_consonant_clusters(text)
        
        # Apply phonetic substitutions
        text = self._apply_phonetic_substitutions(text)
        
        # Generate final phonetic code
        code = self._generate_phonetic_code(text)
        
        return code[:self.config.max_code_length]

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for phonetic encoding."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove punctuation except hyphens and apostrophes
        text = re.sub(r'[^\w\s\-\']', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def _convert_devanagari(self, text: str) -> str:
        """Convert Devanagari script to Roman."""
        result = ""
        for char in text:
            if char in self.devanagari_to_roman:
                result += self.devanagari_to_roman[char]
            else:
                result += char
        return result

    def _normalize_aspiration(self, text: str) -> str:
        """Normalize aspirated consonants."""
        for aspirated, unaspirated in self.aspiration_pairs:
            # Create pattern that handles word boundaries
            pattern = rf'\b{re.escape(aspirated)}\b|\b{re.escape(aspirated)}(?=[aeiou])'
            text = re.sub(pattern, unaspirated, text)
        return text

    def _normalize_vowel_lengths(self, text: str) -> str:
        """Normalize long and short vowels."""
        # Long vowels to short
        vowel_mappings = [
            ('aa', 'a'), ('ii', 'i'), ('uu', 'u'),
            ('oo', 'o'), ('ee', 'e'), ('ai', 'e'), ('au', 'o')
        ]
        
        for long_vowel, short_vowel in vowel_mappings:
            text = text.replace(long_vowel, short_vowel)
        
        # Remove duplicate vowels
        text = re.sub(r'([aeiou])\1+', r'\1', text)
        
        return text

    def _apply_retroflex_mappings(self, text: str) -> str:
        """Apply retroflex to dental mappings."""
        for retroflex, dental in self.retroflex_mappings.items():
            text = text.replace(retroflex, dental)
        return text

    def _apply_sibilant_mappings(self, text: str) -> str:
        """Apply sibilant mappings."""
        for sibilant, replacement in self.sibilant_mappings.items():
            text = text.replace(sibilant, replacement)
        return text

    def _handle_consonant_clusters(self, text: str) -> str:
        """Handle consonant clusters."""
        for cluster, simplified in self.consonant_clusters.items():
            text = text.replace(cluster, simplified)
        return text

    def _apply_phonetic_substitutions(self, text: str) -> str:
        """Apply common phonetic substitutions."""
        for original, substitute in self.phonetic_substitutions:
            # Apply with word boundary consideration
            pattern = rf'\b{re.escape(original)}\b'
            text = re.sub(pattern, substitute, text)
        return text

    def _generate_phonetic_code(self, text: str) -> str:
        """Generate final phonetic code."""
        # Remove spaces and create continuous code
        code = text.replace(' ', '')
        
        # Remove duplicate consecutive characters
        code = re.sub(r'(.)\1+', r'\1', code)
        
        # Ensure minimum length
        if len(code) < 2:
            code += 'x' * (2 - len(code))
        
        return code

    def _soundex_encode(self, text: str) -> str:
        """Traditional Soundex encoding with Sanskrit modifications."""
        text = self._preprocess_text(text)
        
        if not text:
            return "0000"
        
        # Keep first character
        result = text[0].upper()
        
        # Soundex character mappings with Sanskrit modifications
        soundex_map = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1', 'W': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3', 'TH': '3', 'DH': '3',  # Sanskrit aspiration
            'L': '4',
            'M': '5', 'N': '5', 'NG': '5',  # Sanskrit nasals
            'R': '6'
        }
        
        # Process remaining characters
        for i in range(1, len(text)):
            char = text[i].upper()
            
            # Handle digraphs
            if i < len(text) - 1:
                digraph = text[i:i+2].upper()
                if digraph in soundex_map:
                    code = soundex_map[digraph]
                    if len(result) < 4 and (not result or result[-1] != code):
                        result += code
                    continue
            
            if char in soundex_map:
                code = soundex_map[char]
                if len(result) < 4 and (not result or result[-1] != code):
                    result += code
        
        # Pad with zeros
        return result.ljust(4, '0')[:4]

    def _hybrid_encode(self, text: str) -> str:
        """Hybrid encoding combining multiple algorithms."""
        sanskrit_code = self._sanskrit_phonetic_encode(text)
        soundex_code = self._soundex_encode(text)
        
        # Combine codes with weighted preference for Sanskrit encoding
        hybrid_code = sanskrit_code[:4] + soundex_code[:2]
        
        return hybrid_code[:self.config.max_code_length]

    def calculate_phonetic_similarity(self, text1: str, text2: str) -> PhoneticMatch:
        """
        Calculate phonetic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            PhoneticMatch with similarity details
        """
        code1 = self.encode_text(text1)
        code2 = self.encode_text(text2)
        
        # Calculate similarity score using Levenshtein-like approach
        similarity = self._calculate_code_similarity(code1, code2)
        
        # Calculate confidence based on length similarity and code match
        length_factor = 1.0 - abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1)
        confidence = (similarity + length_factor) / 2
        
        return PhoneticMatch(
            original_text=text1,
            target_text=text2,
            original_code=code1,
            target_code=code2,
            similarity_score=similarity,
            algorithm_used=self.config.algorithm,
            confidence=confidence
        )

    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between phonetic codes."""
        if not code1 or not code2:
            return 0.0
        
        if code1 == code2:
            return 1.0
        
        # Use modified Levenshtein for phonetic codes
        max_len = max(len(code1), len(code2))
        min_len = min(len(code1), len(code2))
        
        # Calculate character-level similarity
        matches = 0
        for i in range(min_len):
            if code1[i] == code2[i]:
                matches += 1
        
        # Account for length difference
        base_similarity = matches / max_len
        
        # Bonus for exact matches at start (important for phonetic similarity)
        start_bonus = 0.0
        if len(code1) > 0 and len(code2) > 0 and code1[0] == code2[0]:
            start_bonus = 0.1
        
        return min(1.0, base_similarity + start_bonus)

    def find_phonetic_matches(self, query: str, candidates: List[str], 
                            top_k: int = 10) -> List[PhoneticMatch]:
        """
        Find phonetic matches for a query in a list of candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top matches to return
            
        Returns:
            List of phonetic matches sorted by similarity
        """
        matches = []
        
        for candidate in candidates:
            match = self.calculate_phonetic_similarity(query, candidate)
            if match.similarity_score >= self.config.similarity_threshold:
                matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        
        return matches[:top_k]

    def encode_lexicon_batch(self, lexicon_entries: Dict[str, Any]) -> Dict[str, str]:
        """
        Encode a batch of lexicon entries for efficient matching.
        
        Args:
            lexicon_entries: Dictionary of lexicon entries
            
        Returns:
            Dictionary mapping terms to phonetic codes
        """
        phonetic_codes = {}
        
        for term, entry in lexicon_entries.items():
            # Encode original term
            phonetic_codes[term] = self.encode_text(term)
            
            # Encode variations if available
            if 'variations' in entry:
                for variation in entry['variations']:
                    if variation not in phonetic_codes:
                        phonetic_codes[variation] = self.encode_text(variation)
        
        self.logger.info(f"Generated phonetic codes for {len(phonetic_codes)} terms")
        return phonetic_codes

    def save_phonetic_mappings(self, file_path: Path, 
                              lexicon_codes: Dict[str, str]) -> bool:
        """
        Save phonetic mappings to file.
        
        Args:
            file_path: Path to save mappings
            lexicon_codes: Phonetic codes to save
            
        Returns:
            True if successful
        """
        try:
            mappings_data = {
                'config': {
                    'algorithm': self.config.algorithm.value,
                    'max_code_length': self.config.max_code_length,
                    'similarity_threshold': self.config.similarity_threshold
                },
                'phonetic_codes': lexicon_codes,
                'generated_at': str(Path().cwd())
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(mappings_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved phonetic mappings to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving phonetic mappings: {e}")
            return False

    def load_phonetic_mappings(self, file_path: Path) -> Optional[Dict[str, str]]:
        """
        Load phonetic mappings from file.
        
        Args:
            file_path: Path to mappings file
            
        Returns:
            Phonetic codes dictionary if successful
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                mappings_data = json.load(f)
            
            self.logger.info(f"Loaded phonetic mappings from {file_path}")
            return mappings_data.get('phonetic_codes', {})
            
        except Exception as e:
            self.logger.error(f"Error loading phonetic mappings: {e}")
            return None