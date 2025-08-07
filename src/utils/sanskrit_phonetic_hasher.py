"""
Sanskrit Phonetic Hasher for Story 2.4.3 - Stage 1 of Hybrid Matching Pipeline

This module implements Sanskrit-specific phonetic encoding for fast candidate filtering
in scripture verse identification. Optimized for ASR transcription variations and
Sanskrit linguistic characteristics.

Key Features:
- Sanskrit-specific phonetic encoding algorithm
- Fast hash-based candidate filtering
- IAST transliteration support
- Common ASR error patterns handling
"""

import re
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from utils.logger_config import get_logger
from scripture_processing.canonical_text_manager import VerseCandidate


@dataclass
class PhoneticHashEntry:
    """Entry in the phonetic hash index."""
    verse_id: str
    original_text: str
    phonetic_hash: str
    hash_components: List[str]
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhoneticCandidateMatch:
    """Result of phonetic hash matching."""
    verse_candidate: VerseCandidate
    phonetic_score: float
    hash_distance: int
    original_hash: str
    candidate_hash: str
    matching_components: List[str]


class SanskritPhoneticHasher:
    """
    Sanskrit-specific phonetic hashing system for fast scripture candidate filtering.
    
    This class implements Stage 1 of the hybrid matching pipeline, providing:
    1. Sanskrit-specific phonetic encoding that handles ASR variations
    2. Fast hash-based lookup for initial candidate filtering
    3. IAST transliteration support for consistent encoding
    4. Optimized performance for large scripture databases
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_dir: Optional[Path] = None):
        """
        Initialize the Sanskrit phonetic hasher.
        
        Args:
            config: Configuration parameters
            cache_dir: Directory for caching phonetic hash indexes
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Setup cache directory
        self.cache_dir = cache_dir or Path("data/phonetic_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Phonetic mapping rules for Sanskrit
        self._initialize_phonetic_mappings()
        
        # Hash index for fast lookups
        self.hash_index: Dict[str, List[PhoneticHashEntry]] = {}
        self.verse_hash_map: Dict[str, str] = {}  # verse_id -> hash
        
        # Performance metrics
        self.stats = {
            'hashes_generated': 0,
            'index_lookups': 0,
            'cache_hits': 0,
            'total_indexed_verses': 0
        }
        
        self.logger.info("Sanskrit phonetic hasher initialized")
    
    def _initialize_phonetic_mappings(self) -> None:
        """Initialize Sanskrit-specific phonetic mapping rules."""
        
        # Vowel mappings (including IAST)
        self.vowel_mappings = {
            # Short vowels
            'a': 'A', 'अ': 'A',
            'i': 'I', 'इ': 'I', 'ि': 'I',
            'u': 'U', 'उ': 'U', 'ु': 'U',
            'e': 'E', 'ए': 'E', 'े': 'E',
            'o': 'O', 'ओ': 'O', 'ो': 'O',
            
            # Long vowels (IAST)
            'ā': 'A', 'आ': 'A', 'ा': 'A',
            'ī': 'I', 'ई': 'I', 'ी': 'I',
            'ū': 'U', 'ऊ': 'U', 'ू': 'U',
            
            # Special vowels
            'ṛ': 'R', 'ऋ': 'R', 'ृ': 'R',
            'ṝ': 'R', 'ॠ': 'R',
            'ḷ': 'L', 'ऌ': 'L', 'ॢ': 'L',
        }
        
        # Consonant mappings
        self.consonant_mappings = {
            # Stops (grouped by phonetic similarity)
            'k': 'K', 'ক': 'K', 'ख': 'K', 'kh': 'K',
            'g': 'G', 'ग': 'G', 'घ': 'G', 'gh': 'G',
            'ङ': 'NG', 'ṅ': 'NG',
            
            'c': 'C', 'च': 'C', 'छ': 'C', 'ch': 'C',
            'j': 'J', 'ज': 'J', 'झ': 'J', 'jh': 'J',
            'ञ': 'NY', 'ñ': 'NY',
            
            'ट': 'T', 'ठ': 'T', 'ṭ': 'T', 'ṭh': 'T',
            'ड': 'D', 'ढ': 'D', 'ḍ': 'D', 'ḍh': 'D',
            'ण': 'N', 'ṇ': 'N',
            
            't': 'T', 'त': 'T', 'थ': 'T', 'th': 'T',
            'd': 'D', 'द': 'D', 'ध': 'D', 'dh': 'D',
            'n': 'N', 'न': 'N',
            
            'p': 'P', 'प': 'P', 'फ': 'P', 'ph': 'P',
            'b': 'B', 'ब': 'B', 'भ': 'B', 'bh': 'B',
            'm': 'M', 'म': 'M',
            
            # Approximants and liquids
            'y': 'Y', 'य': 'Y',
            'r': 'R', 'र': 'R',
            'l': 'L', 'ल': 'L',
            'v': 'V', 'व': 'V', 'w': 'V',  # w->v common ASR error
            
            # Sibilants
            'श': 'S', 'ś': 'S', 'sh': 'S',
            'ष': 'S', 'ṣ': 'S',
            's': 'S', 'स': 'S',
            
            # Aspirate
            'h': 'H', 'ह': 'H',
            
            # Anusvara and Visarga
            'ं': 'M', 'ṃ': 'M',
            'ः': 'H', 'ḥ': 'H',
        }
        
        # Common ASR error patterns for Sanskrit
        self.asr_error_patterns = {
            # Aspirated vs non-aspirated confusion
            'kh': 'k', 'gh': 'g', 'ch': 'c', 'jh': 'j',
            'th': 't', 'dh': 'd', 'ph': 'p', 'bh': 'b',
            
            # Retroflex vs dental confusion
            'ṭ': 't', 'ḍ': 'd', 'ṇ': 'n',
            'ṣ': 's', 'ś': 's',
            
            # Long vs short vowel confusion
            'ā': 'a', 'ī': 'i', 'ū': 'u',
            
            # Common mispronunciations
            'ṛ': 'ri', 'ṝ': 'ri',
            'ḷ': 'li',
            
            # Nasalization variations
            'ṃ': 'n', 'ṃ': 'm',
            'ṅ': 'n', 'ñ': 'n',
        }
        
        # Word boundary markers and punctuation to ignore
        self.ignore_patterns = [
            r'[।|]',  # Sanskrit punctuation
            r'[.,!?;:\-\(\)\[\]{}"]',  # Common punctuation  
            r'\s+',   # Whitespace
            r'\d+',   # Numbers
        ]
    
    def generate_phonetic_hash(self, text: str, hash_length: int = 8) -> str:
        """
        Generate Sanskrit-specific phonetic hash for text.
        
        Args:
            text: Sanskrit text (IAST, Devanagari, or mixed)
            hash_length: Length of final hash string
            
        Returns:
            Phonetic hash string for fast comparison
        """
        # Clean and normalize text
        normalized_text = self._normalize_text(text)
        
        # Convert to phonetic representation
        phonetic_repr = self._text_to_phonetic(normalized_text)
        
        # Generate components for more granular matching
        components = self._extract_phonetic_components(phonetic_repr)
        
        # Create hash from phonetic representation
        hash_input = ''.join(sorted(components))  # Sort for order independence
        
        # Generate hash (ensure UTF-8 encoding for Unicode characters)
        try:
            full_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
            phonetic_hash = full_hash[:hash_length]
        except UnicodeEncodeError:
            # Fallback: remove non-ASCII characters and try again
            ascii_input = ''.join(c for c in hash_input if ord(c) < 128)
            full_hash = hashlib.md5(ascii_input.encode('utf-8')).hexdigest()
            phonetic_hash = full_hash[:hash_length]
        
        self.stats['hashes_generated'] += 1
        
        return phonetic_hash
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize Sanskrit text for consistent phonetic processing.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase for consistency
        normalized = text.lower().strip()
        
        # Remove ignored patterns
        for pattern in self.ignore_patterns:
            normalized = re.sub(pattern, '', normalized)
        
        # Handle common ASR errors
        for error, correction in self.asr_error_patterns.items():
            normalized = normalized.replace(error, correction)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _text_to_phonetic(self, text: str) -> str:
        """
        Convert normalized text to phonetic representation.
        
        Args:
            text: Normalized text
            
        Returns:
            Phonetic representation
        """
        phonetic_chars = []
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # Try two-character combinations first
            if i + 1 < len(text):
                two_char = text[i:i+2]
                if two_char in self.consonant_mappings:
                    phonetic_chars.append(self.consonant_mappings[two_char])
                    i += 2
                    continue
                elif two_char in self.vowel_mappings:
                    phonetic_chars.append(self.vowel_mappings[two_char])
                    i += 2
                    continue
            
            # Single character mapping
            if char in self.consonant_mappings:
                phonetic_chars.append(self.consonant_mappings[char])
            elif char in self.vowel_mappings:
                phonetic_chars.append(self.vowel_mappings[char])
            elif char.isalpha():
                # Unknown character - keep as is but uppercase
                phonetic_chars.append(char.upper())
            
            i += 1
        
        return ''.join(phonetic_chars)
    
    def _extract_phonetic_components(self, phonetic_repr: str) -> List[str]:
        """
        Extract phonetic components for granular matching.
        
        Args:
            phonetic_repr: Phonetic representation
            
        Returns:
            List of phonetic components
        """
        components = []
        
        # Split into phonetic syllables/segments
        # This is a simplified approach - could be enhanced with proper syllable detection
        segment_length = 2
        
        for i in range(0, len(phonetic_repr), segment_length):
            segment = phonetic_repr[i:i+segment_length]
            if len(segment) >= 1:  # Include single chars too
                components.append(segment)
        
        # Also include whole representation for exact matching
        components.append(phonetic_repr)
        
        return components
    
    def calculate_hash_distance(self, hash1: str, hash2: str) -> int:
        """
        Calculate distance between two phonetic hashes.
        
        Args:
            hash1: First phonetic hash
            hash2: Second phonetic hash
            
        Returns:
            Distance (0 = identical, higher = more different)
        """
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2))
        
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        return distance
    
    def build_hash_index(self, verses: List[VerseCandidate]) -> Dict[str, Any]:
        """
        Build phonetic hash index from verse candidates.
        
        Args:
            verses: List of verse candidates to index
            
        Returns:
            Index statistics
        """
        self.hash_index.clear()
        self.verse_hash_map.clear()
        
        indexed_count = 0
        
        for verse in verses:
            try:
                # Generate hash for verse canonical text
                phonetic_hash = self.generate_phonetic_hash(verse.canonical_text)
                
                # Create hash entry
                verse_id = f"{verse.source.value}_{verse.chapter}_{verse.verse}"
                
                hash_entry = PhoneticHashEntry(
                    verse_id=verse_id,
                    original_text=verse.canonical_text,
                    phonetic_hash=phonetic_hash,
                    hash_components=self._extract_phonetic_components(
                        self._text_to_phonetic(self._normalize_text(verse.canonical_text))
                    ),
                    generation_metadata={
                        'source': verse.source.value,
                        'chapter': verse.chapter,
                        'verse': verse.verse,
                        'text_length': len(verse.canonical_text)
                    }
                )
                
                # Add to index
                if phonetic_hash not in self.hash_index:
                    self.hash_index[phonetic_hash] = []
                
                self.hash_index[phonetic_hash].append(hash_entry)
                self.verse_hash_map[verse_id] = phonetic_hash
                
                indexed_count += 1
                
            except Exception as e:
                self.logger.error(f"Error indexing verse {verse.source.value} {verse.chapter}:{verse.verse}: {e}")
        
        self.stats['total_indexed_verses'] = indexed_count
        
        index_stats = {
            'verses_indexed': indexed_count,
            'unique_hashes': len(self.hash_index),
            'hash_collisions': sum(len(entries) - 1 for entries in self.hash_index.values() if len(entries) > 1),
            'average_entries_per_hash': indexed_count / max(len(self.hash_index), 1)
        }
        
        self.logger.info(
            f"Built phonetic hash index: {indexed_count} verses, "
            f"{len(self.hash_index)} unique hashes"
        )
        
        return index_stats
    
    def get_phonetic_candidates(
        self, 
        passage: str, 
        max_candidates: int = 50,
        max_distance: int = 2
    ) -> List[PhoneticCandidateMatch]:
        """
        Get scripture candidates using phonetic hash matching.
        
        Args:
            passage: Input passage to match
            max_candidates: Maximum number of candidates to return
            max_distance: Maximum hash distance for matches
            
        Returns:
            List of phonetic candidate matches, sorted by score
        """
        self.stats['index_lookups'] += 1
        
        # Generate hash for input passage
        passage_hash = self.generate_phonetic_hash(passage)
        
        candidates = []
        
        # Look for exact hash matches first
        if passage_hash in self.hash_index:
            self.stats['cache_hits'] += 1
            for entry in self.hash_index[passage_hash]:
                match = PhoneticCandidateMatch(
                    verse_candidate=self._hash_entry_to_verse_candidate(entry),
                    phonetic_score=1.0,  # Perfect match
                    hash_distance=0,
                    original_hash=passage_hash,
                    candidate_hash=entry.phonetic_hash,
                    matching_components=entry.hash_components
                )
                candidates.append(match)
        
        # Look for near matches within distance threshold
        for candidate_hash, entries in self.hash_index.items():
            distance = self.calculate_hash_distance(passage_hash, candidate_hash)
            
            if 0 < distance <= max_distance:
                # Calculate phonetic score based on distance
                score = 1.0 - (distance / len(passage_hash))
                
                for entry in entries:
                    match = PhoneticCandidateMatch(
                        verse_candidate=self._hash_entry_to_verse_candidate(entry),
                        phonetic_score=max(0.0, score),
                        hash_distance=distance,
                        original_hash=passage_hash,
                        candidate_hash=candidate_hash,
                        matching_components=entry.hash_components
                    )
                    candidates.append(match)
        
        # Sort by phonetic score (descending)
        candidates.sort(key=lambda x: x.phonetic_score, reverse=True)
        
        # Return top candidates
        return candidates[:max_candidates]
    
    def _hash_entry_to_verse_candidate(self, entry: PhoneticHashEntry) -> VerseCandidate:
        """Convert hash entry back to VerseCandidate."""
        from scripture_processing.canonical_text_manager import ScriptureSource
        
        # Parse verse ID
        parts = entry.verse_id.split('_')
        source_str = parts[0]
        chapter = int(parts[1])
        verse = int(parts[2])
        
        # Create VerseCandidate
        return VerseCandidate(
            source=ScriptureSource(source_str),
            chapter=chapter,
            verse=verse,
            canonical_text=entry.original_text,
            confidence_score=0.8,  # Default for phonetic matches
            match_strength="phonetic",
            metadata=entry.generation_metadata
        )
    
    def save_hash_index(self, file_path: Optional[Path] = None) -> None:
        """
        Save phonetic hash index to file for persistence.
        
        Args:
            file_path: Optional custom file path
        """
        if file_path is None:
            file_path = self.cache_dir / "phonetic_hash_index.json"
        
        try:
            # Convert index to serializable format
            serializable_index = {}
            
            for hash_key, entries in self.hash_index.items():
                serializable_index[hash_key] = [
                    {
                        'verse_id': entry.verse_id,
                        'original_text': entry.original_text,
                        'phonetic_hash': entry.phonetic_hash,
                        'hash_components': entry.hash_components,
                        'generation_metadata': entry.generation_metadata
                    }
                    for entry in entries
                ]
            
            index_data = {
                'hash_index': serializable_index,
                'verse_hash_map': self.verse_hash_map,
                'stats': self.stats,
                'metadata': {
                    'version': '2.4.3',
                    'total_verses': self.stats['total_indexed_verses'],
                    'unique_hashes': len(self.hash_index)
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved phonetic hash index to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving hash index: {e}")
    
    def load_hash_index(self, file_path: Optional[Path] = None) -> bool:
        """
        Load phonetic hash index from file.
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            True if loaded successfully
        """
        if file_path is None:
            file_path = self.cache_dir / "phonetic_hash_index.json"
        
        if not file_path.exists():
            self.logger.info("No existing phonetic hash index found")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Reconstruct hash index
            self.hash_index.clear()
            for hash_key, entries_data in index_data['hash_index'].items():
                entries = []
                for entry_data in entries_data:
                    entry = PhoneticHashEntry(**entry_data)
                    entries.append(entry)
                self.hash_index[hash_key] = entries
            
            # Restore other data
            self.verse_hash_map = index_data['verse_hash_map']
            self.stats.update(index_data['stats'])
            
            self.logger.info(
                f"Loaded phonetic hash index: {len(self.hash_index)} hashes, "
                f"{self.stats['total_indexed_verses']} verses"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading hash index: {e}")
            return False
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'phonetic_hashing': {
                'hashes_generated': self.stats['hashes_generated'],
                'index_lookups': self.stats['index_lookups'],
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': f"{(self.stats['cache_hits'] / max(self.stats['index_lookups'], 1)) * 100:.1f}%"
            },
            'index_status': {
                'total_indexed_verses': self.stats['total_indexed_verses'],
                'unique_hashes': len(self.hash_index),
                'memory_usage_entries': sum(len(entries) for entries in self.hash_index.values())
            }
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate phonetic hasher configuration."""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check cache directory
        if not self.cache_dir.exists():
            validation['errors'].append(f"Cache directory missing: {self.cache_dir}")
            validation['is_valid'] = False
        
        # Check index status
        if not self.hash_index:
            validation['warnings'].append("No phonetic hash index loaded - call build_hash_index() first")
        
        # Check phonetic mappings
        if not self.consonant_mappings or not self.vowel_mappings:
            validation['errors'].append("Phonetic mappings not initialized")
            validation['is_valid'] = False
        
        return validation