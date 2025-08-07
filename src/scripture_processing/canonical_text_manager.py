"""
Canonical Text Manager Module.

This module manages canonical scriptural texts and provides lookup capabilities
for verse substitution operations.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from utils.logger_config import get_logger
from sanskrit_hindi_identifier.word_identifier import LexiconEntry


class ScriptureSource(Enum):
    """Supported scriptural sources."""
    BHAGAVAD_GITA = "bhagavad_gita"
    UPANISHADS = "upanishads" 
    YOGA_SUTRAS = "yoga_sutras"
    VEDAS = "vedas"
    PURANAS = "puranas"


@dataclass
class CanonicalVerse:
    """Represents a canonical verse with complete metadata."""
    id: str
    source: ScriptureSource
    chapter: int
    verse: int
    canonical_text: str
    transliteration: str
    translation: Optional[str]
    commentary: Optional[str]
    source_authority: str
    tags: List[str]
    variations: List[str]


@dataclass
class VerseReference:
    """Reference to a specific verse location."""
    source: ScriptureSource
    chapter: int
    verse: int
    
    def __str__(self) -> str:
        return f"{self.source.value} {self.chapter}.{self.verse}"


class CanonicalTextManager:
    """
    Manager for canonical scriptural texts and verse lookup.
    
    Provides comprehensive database of canonical verses with lookup capabilities,
    metadata management, and integration with lexicon systems.
    """
    
    def __init__(self, scripture_dir: Path = None, config: Dict = None):
        """
        Initialize the Canonical Text Manager.
        
        Args:
            scripture_dir: Directory containing scripture databases
            config: Configuration parameters
        """
        self.logger = get_logger(__name__)
        self.scripture_dir = scripture_dir or Path("data/scriptures")
        self.config = config or {}
        
        # Data structures
        self.canonical_verses: Dict[str, CanonicalVerse] = {}
        self.source_indexes: Dict[ScriptureSource, Dict[str, CanonicalVerse]] = {}
        self.text_indexes: Dict[str, List[CanonicalVerse]] = {}
        self.variation_indexes: Dict[str, CanonicalVerse] = {}
        
        # Initialize indexes
        for source in ScriptureSource:
            self.source_indexes[source] = {}
        
        # Load canonical texts
        self._load_canonical_texts()
        
        self.logger.info(f"Loaded {len(self.canonical_verses)} canonical verses from {len(ScriptureSource)} sources")
    
    def _load_canonical_texts(self) -> None:
        """Load canonical texts from scripture database files."""
        scripture_files = {
            ScriptureSource.BHAGAVAD_GITA: "bhagavad_gita.yaml",
            ScriptureSource.UPANISHADS: "upanishads.yaml",
            ScriptureSource.YOGA_SUTRAS: "yoga_sutras.yaml"
        }
        
        for source, filename in scripture_files.items():
            file_path = self.scripture_dir / filename
            if file_path.exists():
                self._load_scripture_file(file_path, source)
            else:
                self.logger.warning(f"Scripture file not found: {file_path}")
                # Create sample file structure
                self._create_sample_scripture_file(file_path, source)
    
    def _load_scripture_file(self, file_path: Path, source: ScriptureSource) -> None:
        """Load a single scripture file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if 'verses' not in data:
                self.logger.warning(f"No 'verses' section found in {file_path}")
                return
            
            verses_loaded = 0
            for verse_data in data['verses']:
                verse = self._create_canonical_verse(verse_data, source)
                if verse:
                    self._index_verse(verse)
                    verses_loaded += 1
            
            self.logger.info(f"Loaded {verses_loaded} verses from {source.value}")
            
        except Exception as e:
            self.logger.error(f"Error loading scripture file {file_path}: {e}")
    
    def _create_canonical_verse(self, verse_data: Dict, source: ScriptureSource) -> Optional[CanonicalVerse]:
        """Create a CanonicalVerse from data."""
        try:
            verse_id = f"{source.value}_{verse_data.get('chapter', 0)}_{verse_data.get('verse', 0)}"
            
            verse = CanonicalVerse(
                id=verse_id,
                source=source,
                chapter=verse_data.get('chapter', 0),
                verse=verse_data.get('verse', 0),
                canonical_text=verse_data.get('canonical_text', ''),
                transliteration=verse_data.get('transliteration', ''),
                translation=verse_data.get('translation'),
                commentary=verse_data.get('commentary'),
                source_authority=verse_data.get('source_authority', 'IAST'),
                tags=verse_data.get('tags', []),
                variations=verse_data.get('variations', [])
            )
            
            return verse
            
        except Exception as e:
            self.logger.error(f"Error creating canonical verse: {e}")
            return None
    
    def _index_verse(self, verse: CanonicalVerse) -> None:
        """Index a canonical verse for efficient lookup."""
        # Main index
        self.canonical_verses[verse.id] = verse
        
        # Source index
        chapter_verse_key = f"{verse.chapter}.{verse.verse}"
        self.source_indexes[verse.source][chapter_verse_key] = verse
        
        # Text index (by words)
        words = verse.canonical_text.lower().split()
        for i in range(len(words)):
            for j in range(i+1, min(i+6, len(words)+1)):  # Index 2-5 word phrases
                phrase = ' '.join(words[i:j])
                if phrase not in self.text_indexes:
                    self.text_indexes[phrase] = []
                self.text_indexes[phrase].append(verse)
        
        # Variation index
        for variation in verse.variations:
            self.variation_indexes[variation.lower()] = verse
        
        # Also index transliteration
        if verse.transliteration:
            self.variation_indexes[verse.transliteration.lower()] = verse
    
    def lookup_verse_by_reference(self, reference: VerseReference) -> Optional[CanonicalVerse]:
        """
        Look up a verse by its reference.
        
        Args:
            reference: Verse reference
            
        Returns:
            Canonical verse if found
        """
        chapter_verse_key = f"{reference.chapter}.{reference.verse}"
        return self.source_indexes[reference.source].get(chapter_verse_key)
    
    def lookup_verse_by_text(self, text: str, min_words: int = 3) -> List[CanonicalVerse]:
        """
        Look up verses by partial text match.
        
        Args:
            text: Text to search for
            min_words: Minimum word overlap required
            
        Returns:
            List of matching canonical verses
        """
        text_lower = text.lower().strip()
        matches = []
        
        # Direct phrase lookup
        words = text_lower.split()
        for i in range(len(words)):
            for j in range(i+min_words, min(i+6, len(words)+1)):
                phrase = ' '.join(words[i:j])
                if phrase in self.text_indexes:
                    matches.extend(self.text_indexes[phrase])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for verse in matches:
            if verse.id not in seen:
                seen.add(verse.id)
                unique_matches.append(verse)
        
        return unique_matches
    
    def lookup_verse_by_variation(self, variation: str) -> Optional[CanonicalVerse]:
        """
        Look up a verse by its variation or transliteration.
        
        Args:
            variation: Variation text to search for
            
        Returns:
            Canonical verse if found
        """
        return self.variation_indexes.get(variation.lower())
    
    def search_verses_by_content(self, query: str, limit: int = 10) -> List[CanonicalVerse]:
        """
        Search verses by content using fuzzy matching.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching verses
        """
        query_lower = query.lower()
        scored_matches = []
        
        for verse in self.canonical_verses.values():
            # Calculate simple relevance score
            canonical_lower = verse.canonical_text.lower()
            transliteration_lower = verse.transliteration.lower()
            
            # Word overlap scoring
            query_words = set(query_lower.split())
            canonical_words = set(canonical_lower.split())
            transliteration_words = set(transliteration_lower.split())
            
            canonical_overlap = len(query_words & canonical_words) / max(len(query_words), 1)
            transliteration_overlap = len(query_words & transliteration_words) / max(len(query_words), 1)
            
            # Simple substring scoring
            canonical_substring = query_lower in canonical_lower
            transliteration_substring = query_lower in transliteration_lower
            
            # Combined score
            score = (canonical_overlap * 0.4 + 
                    transliteration_overlap * 0.3 +
                    (0.2 if canonical_substring else 0.0) +
                    (0.1 if transliteration_substring else 0.0))
            
            if score > 0:
                scored_matches.append((score, verse))
        
        # Sort by score and return top matches
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        return [verse for score, verse in scored_matches[:limit]]
    
    def get_verse_candidates(self, text: str, source: Optional[ScriptureSource] = None,
                           max_candidates: int = 5) -> List[CanonicalVerse]:
        """
        Get potential verse candidates for a given text.
        
        Args:
            text: Input text to match
            source: Specific scripture source to search (optional)
            max_candidates: Maximum candidates to return
            
        Returns:
            List of candidate verses with relevance scoring
        """
        candidates = []
        
        # Text-based lookup
        text_matches = self.lookup_verse_by_text(text)
        candidates.extend(text_matches)
        
        # Variation lookup
        variation_match = self.lookup_verse_by_variation(text)
        if variation_match:
            candidates.append(variation_match)
        
        # Content search
        content_matches = self.search_verses_by_content(text, limit=max_candidates)
        candidates.extend(content_matches)
        
        # Filter by source if specified
        if source:
            candidates = [v for v in candidates if v.source == source]
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for verse in candidates:
            if verse.id not in seen:
                seen.add(verse.id)
                unique_candidates.append(verse)
        
        return unique_candidates[:max_candidates]
    
    def get_verse_context(self, verse: CanonicalVerse, context_verses: int = 2) -> List[CanonicalVerse]:
        """
        Get contextual verses around a given verse.
        
        Args:
            verse: Target verse
            context_verses: Number of verses before/after to include
            
        Returns:
            List of verses including context
        """
        context = []
        source_verses = list(self.source_indexes[verse.source].values())
        source_verses.sort(key=lambda v: (v.chapter, v.verse))
        
        try:
            verse_index = source_verses.index(verse)
            start_idx = max(0, verse_index - context_verses)
            end_idx = min(len(source_verses), verse_index + context_verses + 1)
            context = source_verses[start_idx:end_idx]
        except ValueError:
            # If verse not found in sorted list, just return the verse itself
            context = [verse]
        
        return context
    
    def _create_sample_scripture_file(self, file_path: Path, source: ScriptureSource) -> None:
        """Create a sample scripture file if it doesn't exist."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        sample_data = {
            'metadata': {
                'source': source.value,
                'version': '1.0',
                'authority': 'IAST',
                'description': f'Canonical verses from {source.value}',
                'created': datetime.now().isoformat()
            },
            'verses': [
                {
                    'chapter': 2,
                    'verse': 25,
                    'canonical_text': "avyakto 'yam acintyo 'yam avikāryo 'yam ucyate | tasmād evaṃ viditvainaṃ nānuśocitum arhasi ||",
                    'transliteration': "avyakto 'yam acintyo 'yam avikāryo 'yam ucyate | tasmād evaṃ viditvainaṃ nānuśocitum arhasi ||",
                    'translation': "This soul is said to be unmanifest, unthinkable and unchanging. Therefore, knowing this, you should not grieve.",
                    'commentary': "Sample commentary for demonstration",
                    'source_authority': 'IAST',
                    'tags': ['soul', 'unchanging', 'grief'],
                    'variations': [f"{source.value} 2.25", f"{source.value} chapter 2 verse 25"]
                }
            ]
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(sample_data, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"Created sample scripture file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error creating sample scripture file {file_path}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the canonical text database."""
        stats = {
            'total_verses': len(self.canonical_verses),
            'sources': {source.value: len(index) for source, index in self.source_indexes.items()},
            'text_index_entries': len(self.text_indexes),
            'variation_entries': len(self.variation_indexes)
        }
        
        # Chapter/verse distribution by source
        for source, verses in self.source_indexes.items():
            chapters = set()
            for verse in verses.values():
                chapters.add(verse.chapter)
            stats['sources'][f'{source.value}_chapters'] = len(chapters)
        
        return stats