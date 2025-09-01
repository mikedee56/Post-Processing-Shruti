"""
ASR Scripture Matcher - Digital Dharma Implementation
Matches garbled ASR output to canonical scriptural verses using multi-stage pipeline
Based on research insights from Digital Dharma paper
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import json
from difflib import SequenceMatcher
import Levenshtein
from collections import defaultdict
import unicodedata

# Configure logging
logger = logging.getLogger(__name__)

class MatchingStrategy(Enum):
    """Matching strategies based on Digital Dharma research"""
    PHONETIC = "phonetic"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    ABBREVIATION = "abbreviation"

@dataclass
class ASRMatch:
    """Represents a match between ASR text and canonical verse"""
    asr_text: str
    canonical_verse: Dict[str, Any]
    confidence_score: float
    matching_strategy: MatchingStrategy
    match_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def verse_reference(self) -> str:
        """Get formatted verse reference"""
        verse = self.canonical_verse
        source = verse.get('source', 'Unknown')
        chapter = verse.get('chapter', '?')
        verse_num = verse.get('verse', '?')
        return f"{source} {chapter}.{verse_num}"
    
    @property
    def canonical_text(self) -> str:
        """Get canonical IAST text"""
        return self.canonical_verse.get('canonical_text', '')

class SanskritPhoneticEncoder:
    """
    Sanskrit-aware phonetic encoding for matching ASR output
    Handles common ASR mistakes in Sanskrit pronunciation
    """
    
    def __init__(self):
        # Common ASR mistakes in Sanskrit
        self.asr_substitutions = {
            # Vowel confusions
            'a': ['uh', 'ah', 'aa'],
            'ā': ['aa', 'ah', 'a'],
            'i': ['ee', 'ih', 'e'],
            'ī': ['ee', 'i', 'eeh'],
            'u': ['oo', 'uh', 'o'],
            'ū': ['oo', 'u', 'ooh'],
            
            # Consonant confusions
            'ṛ': ['ri', 'ru', 'r', 'rr'],
            'ṇ': ['n', 'nn', 'na'],
            'ñ': ['ny', 'n', 'gn'],
            'ṭ': ['t', 'tt', 'ta'],
            'ḍ': ['d', 'dd', 'da'],
            'ś': ['sh', 's', 'sch'],
            'ṣ': ['sh', 's', 'shh'],
            'kṣ': ['ksh', 'ks', 'x'],
            'jñ': ['gn', 'gy', 'jn', 'gya'],
            
            # Common word confusions
            'dharma': ['dharm', 'dharama', 'darma', 'dhurma'],
            'karma': ['karm', 'kurma', 'karama'],
            'yoga': ['yog', 'yowga', 'yogah'],
            'bhagavad': ['bhagwad', 'bhagvad', 'bhagawad', 'bagavad'],
            'gītā': ['gita', 'geeta', 'gitta'],
            'kṛṣṇa': ['krishna', 'krsna', 'krshna', 'krishn'],
            'arjuna': ['arjun', 'urjuna', 'arjunah'],
            
            # Ramayana-specific terms (Hindi Tulsi Ramayana)
            'राम': ['ram', 'rama', 'raam', 'rahm'],
            'सीता': ['sita', 'seeta', 'sitta', 'seta'],
            'हनुमान': ['hanuman', 'hanumat', 'hanumant', 'hunuman'],
            'रामायण': ['ramayana', 'ramayan', 'ramayane', 'ramayne'],
            'तुलसी': ['tulsi', 'tulasi', 'tulsee', 'tulashi'],
            'दशरथ': ['dasrath', 'dasaratha', 'dashrath', 'dashratha'],
            'जानकी': ['janki', 'janaki', 'jankee', 'jaanki'],
            'रघुबर': ['raghubar', 'raghubara', 'raghuver', 'raghuvir'],
            'अयोध्या': ['ayodhya', 'ayodya', 'ayodhiya', 'ayudhya'],
            'लंका': ['lanka', 'lanka', 'lanke', 'lunka'],
            'रावण': ['ravan', 'ravana', 'rawon', 'ravaan'],
            'लक्ष्मण': ['lakshman', 'laxman', 'laksman', 'lukshmun'],
            'भरत': ['bharat', 'bharata', 'bhurt', 'bharath'],
            'शत्रुघ्न': ['shatrughna', 'shatrughn', 'shatrugun', 'shtrughn'],
            
            # Hindi Ramayana terms
            'चरण': ['charan', 'charun', 'churn', 'chaaran'],
            'सरोज': ['saroj', 'sarooja', 'suroj', 'sarauj'],
            'गुरु': ['guru', 'gur', 'guroo', 'gurun'],
            'मंगल': ['mangal', 'mangul', 'mungal', 'mangaal'],
            'भवन': ['bhavan', 'bhvn', 'bhuwun', 'bhwan'],
            'अमंगल': ['amangal', 'umangal', 'amungal', 'amangaal'],
            'हारी': ['hari', 'haari', 'haaree', 'hurree'],
            'द्रवहु': ['dravu', 'drawo', 'druvhu', 'drawahu'],
            'बिहारी': ['bihari', 'vihaaree', 'bihaaree', 'behari'],
            'अनंत': ['anant', 'ananta', 'ununt', 'anunt'],
            'कथा': ['katha', 'katha', 'kuttha', 'kaatha'],
            'संता': ['santa', 'santaa', 'sunta', 'saantaa'],
        }
        
        # Build reverse mapping for quick lookup
        self.reverse_map = defaultdict(list)
        for canonical, variants in self.asr_substitutions.items():
            for variant in variants:
                self.reverse_map[variant].append(canonical)
    
    def encode(self, text: str) -> str:
        """
        Create phonetic encoding of Sanskrit text
        Normalizes common ASR variations
        """
        # Lowercase and remove diacritics for base comparison
        text = text.lower().strip()
        
        # Apply common substitutions
        encoded = text
        for variant, canonicals in self.reverse_map.items():
            if variant in encoded:
                # Use the first canonical form
                encoded = encoded.replace(variant, canonicals[0])
        
        # Remove spaces and punctuation for pure phonetic comparison
        encoded = re.sub(r'[^\w\s]', '', encoded)
        encoded = re.sub(r'\s+', '', encoded)
        
        return encoded
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate phonetic similarity between two Sanskrit texts"""
        enc1 = self.encode(text1)
        enc2 = self.encode(text2)
        
        # Use Levenshtein ratio for similarity
        return Levenshtein.ratio(enc1, enc2)

class ASRScriptureMatcher:
    """
    Main class for matching ASR output to canonical scriptural verses
    Implements multi-stage pipeline from Digital Dharma research
    """
    
    def __init__(self, scripture_data_path: Path = None):
        """
        Initialize the ASR Scripture Matcher
        
        Args:
            scripture_data_path: Path to scripture data directory
        """
        self.scripture_data_path = scripture_data_path or Path("data/scriptures")
        self.phonetic_encoder = SanskritPhoneticEncoder()
        self.canonical_verses = self._load_canonical_verses()
        self.abbreviation_patterns = self._build_abbreviation_patterns()
        
        # Cache for performance
        self._phonetic_cache = {}
        self._fuzzy_cache = {}
        
        # Count verses by source for informative logging
        sources = {}
        for verse in self.canonical_verses:
            source = verse.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        logger.info(f"Loaded {len(self.canonical_verses)} canonical verses from {len(sources)} sources: {dict(sources)}")
    
    def _load_canonical_verses(self) -> List[Dict[str, Any]]:
        """Load all canonical verses from comprehensive scripture files"""
        verses = []
        
        # Load Comprehensive Bhagavad Gita (700+ verses)
        gita_comprehensive_path = self.scripture_data_path / "bhagavad_gita_comprehensive.yaml"
        if gita_comprehensive_path.exists():
            with open(gita_comprehensive_path, 'r', encoding='utf-8') as f:
                gita_data = yaml.safe_load(f)
                for verse in gita_data.get('verses', []):
                    verse['source'] = 'Bhagavad Gita'
                    verses.append(verse)
        else:
            # Fallback to original minimal database
            gita_path = self.scripture_data_path / "bhagavad_gita.yaml"
            if gita_path.exists():
                with open(gita_path, 'r', encoding='utf-8') as f:
                    gita_data = yaml.safe_load(f)
                    for verse in gita_data.get('verses', []):
                        verse['source'] = 'Bhagavad Gita'
                        verses.append(verse)
        
        # Load Comprehensive Ramayana (Tulsi Ramayana in Hindi - 300+ verses)
        ramayana_comprehensive_path = self.scripture_data_path / "ramayana_comprehensive.yaml"
        if ramayana_comprehensive_path.exists():
            with open(ramayana_comprehensive_path, 'r', encoding='utf-8') as f:
                ramayana_data = yaml.safe_load(f)
                for verse in ramayana_data.get('verses', []):
                    verse['source'] = 'Ramayana'
                    verses.append(verse)
        else:
            # Fallback to original minimal database
            ramayana_path = self.scripture_data_path / "ramayana.yaml"
            if ramayana_path.exists():
                with open(ramayana_path, 'r', encoding='utf-8') as f:
                    ramayana_data = yaml.safe_load(f)
                    for verse in ramayana_data.get('verses', []):
                        verse['source'] = 'Ramayana'
                        verses.append(verse)
        
        # Load Comprehensive Upanishads (150+ mantras from 10 principal Upanishads)
        upanishads_comprehensive_path = self.scripture_data_path / "upanishads_comprehensive.yaml"
        if upanishads_comprehensive_path.exists():
            with open(upanishads_comprehensive_path, 'r', encoding='utf-8') as f:
                upanishad_data = yaml.safe_load(f)
                for verse in upanishad_data.get('verses', []):
                    verse['source'] = verse.get('upanishad', 'Upanishad')
                    verses.append(verse)
        else:
            # Fallback to original minimal database
            upanishad_path = self.scripture_data_path / "upanishads.yaml"
            if upanishad_path.exists():
                with open(upanishad_path, 'r', encoding='utf-8') as f:
                    upanishad_data = yaml.safe_load(f)
                    for verse in upanishad_data.get('verses', []):
                        verse['source'] = verse.get('upanishad', 'Upanishad')
                        verses.append(verse)
        
        # Load Comprehensive Yoga Sutras (all 196 sutras)
        yoga_sutras_comprehensive_path = self.scripture_data_path / "yoga_sutras_comprehensive.yaml"
        if yoga_sutras_comprehensive_path.exists():
            with open(yoga_sutras_comprehensive_path, 'r', encoding='utf-8') as f:
                sutras_data = yaml.safe_load(f)
                for verse in sutras_data.get('verses', []):
                    verse['source'] = 'Yoga Sutras'
                    verses.append(verse)
        else:
            # Fallback to original minimal database
            sutras_path = self.scripture_data_path / "yoga_sutras.yaml"
            if sutras_path.exists():
                with open(sutras_path, 'r', encoding='utf-8') as f:
                    sutras_data = yaml.safe_load(f)
                    for verse in sutras_data.get('verses', []):
                        verse['source'] = 'Yoga Sutras'
                        verses.append(verse)
        
        return verses
    
    def _build_abbreviation_patterns(self) -> List[Tuple[re.Pattern, Dict[str, Any]]]:
        """Build regex patterns for abbreviated scripture references"""
        patterns = []
        
        # Bhagavad Gita patterns
        gita_patterns = [
            r'(?:bg|gita|geeta|bhagavad\s*gita?)\s*(?:ch(?:apter)?|c)?\s*(\d+)[,.\s]+(?:v(?:erse)?|shloka?)?\s*(\d+)',
            r'(?:chapter|ch)\s*(\d+)\s*(?:verse|v)\s*(\d+)\s*(?:of\s*)?(?:the\s*)?(?:gita|geeta|bhagavad)',
            r'(\d+)[.:](\d+)\s*(?:bg|gita|bhagavad)',
        ]
        
        for pattern_str in gita_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            patterns.append((pattern, {'source': 'Bhagavad Gita'}))
        
        # Upanishad patterns
        upanishad_patterns = [
            r'([\w]+)\s*upanishad\s*(?:ch(?:apter)?|c)?\s*(\d+)[,.\s]+(?:v(?:erse)?|mantra?)?\s*(\d+)',
        ]
        
        for pattern_str in upanishad_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            patterns.append((pattern, {'source': 'Upanishad'}))
        
        return patterns
    
    def match_asr_to_verse(self, asr_text: str, min_confidence: float = 0.3, max_results: int = 5) -> List[ASRMatch]:
        """
        Match ASR text to canonical verses using multiple strategies
        
        Args:
            asr_text: The ASR-transcribed text to match
            min_confidence: Minimum confidence threshold (0.0-1.0)
            max_results: Maximum number of results to return
            
        Returns:
            List of ASRMatch objects sorted by confidence
        """
        if not asr_text.strip():
            return []
        
        all_matches = []
        
        # Strategy 1: Abbreviation matching (highest confidence for exact patterns)
        abbreviation_matches = self._match_abbreviated_reference(asr_text)
        all_matches.extend(abbreviation_matches)
        
        # Strategy 2: Phonetic matching for Sanskrit/Hindi content
        phonetic_matches = self._match_phonetic(asr_text, min_confidence)
        all_matches.extend(phonetic_matches)
        
        # Strategy 3: Fuzzy string matching for variations
        fuzzy_matches = self._match_fuzzy(asr_text, min_confidence)
        all_matches.extend(fuzzy_matches)
        
        # Remove duplicates and sort by confidence
        unique_matches = {}
        for match in all_matches:
            key = f"{match.verse_reference}_{match.matching_strategy.value}"
            if key not in unique_matches or match.confidence_score > unique_matches[key].confidence_score:
                unique_matches[key] = match
        
        # Sort by confidence descending
        sorted_matches = sorted(unique_matches.values(), key=lambda x: x.confidence_score, reverse=True)
        
        # Filter by minimum confidence
        filtered_matches = [m for m in sorted_matches if m.confidence_score >= min_confidence]
        
        return filtered_matches[:max_results]
    
    def _match_abbreviated_reference(self, text: str) -> List[ASRMatch]:
        """Match abbreviated scripture references like 'Gita ch 3, verse 44'"""
        matches = []
        
        for pattern, metadata in self.abbreviation_patterns:
            for match in pattern.finditer(text):
                try:
                    groups = match.groups()
                    
                    if len(groups) >= 2:
                        # Extract chapter and verse numbers
                        chapter = int(groups[-2]) if groups[-2].isdigit() else None
                        verse = int(groups[-1]) if groups[-1].isdigit() else None
                        
                        if chapter and verse:
                            # Find the canonical verse
                            for canonical_verse in self.canonical_verses:
                                if (canonical_verse.get('chapter') == chapter and 
                                    canonical_verse.get('verse') == verse and
                                    metadata['source'] in canonical_verse.get('source', '')):
                                    
                                    asr_match = ASRMatch(
                                        asr_text=match.group(0),
                                        canonical_verse=canonical_verse,
                                        confidence_score=0.95,  # High confidence for exact reference
                                        matching_strategy=MatchingStrategy.ABBREVIATION,
                                        match_details={
                                            'pattern': pattern.pattern,
                                            'chapter': chapter,
                                            'verse': verse
                                        }
                                    )
                                    matches.append(asr_match)
                                    break
                except (AttributeError, IndexError, ValueError) as e:
                    # Log warning but continue processing
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Failed to process match groups in abbreviated reference: {str(e)}")
                    continue
        
        return matches
    
    def _match_phonetic(self, text: str, min_confidence: float = 0.3, top_k: int = 5) -> List[ASRMatch]:
        """Match using Sanskrit phonetic encoding"""
        matches = []
        
        # Cache key for performance
        cache_key = f"phonetic_{text}"
        if cache_key in self._phonetic_cache:
            return self._phonetic_cache[cache_key]
        
        # Calculate phonetic similarity with all verses
        similarities = []
        for verse in self.canonical_verses:
            canonical_text = verse.get('canonical_text', '')
            if canonical_text:
                similarity = self.phonetic_encoder.similarity(text, canonical_text)
                similarities.append((similarity, verse))
        
        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        for similarity, verse in similarities[:top_k]:
            if similarity > min_confidence:  # Use minimum confidence threshold
                asr_match = ASRMatch(
                    asr_text=text,
                    canonical_verse=verse,
                    confidence_score=similarity * 0.8,  # Scale confidence
                    matching_strategy=MatchingStrategy.PHONETIC,
                    match_details={
                        'phonetic_similarity': similarity,
                        'encoded_asr': self.phonetic_encoder.encode(text),
                        'encoded_canonical': self.phonetic_encoder.encode(verse.get('canonical_text', ''))
                    }
                )
                matches.append(asr_match)
        
        self._phonetic_cache[cache_key] = matches
        return matches
    
    def _match_fuzzy(self, text: str, min_confidence: float = 0.3, top_k: int = 5) -> List[ASRMatch]:
        """Match using fuzzy string matching with Levenshtein distance"""
        matches = []
        
        # Cache key for performance
        cache_key = f"fuzzy_{text}"
        if cache_key in self._fuzzy_cache:
            return self._fuzzy_cache[cache_key]
        
        # Calculate Levenshtein distance with all verses
        distances = []
        for verse in self.canonical_verses:
            canonical_text = verse.get('canonical_text', '')
            transliteration = verse.get('transliteration', '')
            
            # Try matching against both canonical text and transliteration
            for target in [canonical_text, transliteration]:
                if target:
                    # Normalize for comparison
                    normalized_text = self._normalize_for_fuzzy(text)
                    normalized_target = self._normalize_for_fuzzy(target)
                    
                    # Calculate similarity
                    distance = Levenshtein.distance(normalized_text, normalized_target)
                    max_len = max(len(normalized_text), len(normalized_target))
                    similarity = 1.0 - (distance / max_len) if max_len > 0 else 0
                    
                    distances.append((similarity, verse, target))
        
        # Sort by similarity and take top_k
        distances.sort(key=lambda x: x[0], reverse=True)
        
        for similarity, verse, matched_text in distances[:top_k]:
            if similarity >= min_confidence:  # Use min_confidence parameter
                asr_match = ASRMatch(
                    asr_text=text,
                    canonical_verse=verse,
                    confidence_score=similarity * 0.7,  # Scale confidence
                    matching_strategy=MatchingStrategy.FUZZY,
                    match_details={
                        'fuzzy_similarity': similarity,
                        'matched_against': matched_text[:50] + '...' if len(matched_text) > 50 else matched_text,
                        'levenshtein_distance': Levenshtein.distance(text, matched_text)
                    }
                )
                matches.append(asr_match)
        
        self._fuzzy_cache[cache_key] = matches
        return matches
    
    def _match_hybrid(self, text: str) -> List[ASRMatch]:
        """
        Hybrid matching combining multiple strategies
        Implements the 3-stage pipeline from Digital Dharma research
        """
        matches = []
        
        # Stage 1: Phonetic filtering (broad search)
        phonetic_candidates = self._match_phonetic(text, top_k=10)
        
        # Stage 2: Sequence alignment (refine candidates)
        sequence_scores = {}
        for i, candidate in enumerate(phonetic_candidates):
            canonical_text = candidate.canonical_verse.get('canonical_text', '')
            if canonical_text:
                # Use SequenceMatcher for alignment
                matcher = SequenceMatcher(None, text.lower(), canonical_text.lower())
                sequence_scores[i] = matcher.ratio()
        
        # Stage 3: Confidence scoring (combine signals)
        for i, candidate in enumerate(phonetic_candidates):
            phonetic_score = candidate.confidence_score
            sequence_score = sequence_scores.get(i, 0)
            
            # Check for keyword matches
            keyword_score = self._calculate_keyword_score(text, candidate.canonical_verse)
            
            # Combine scores with weights
            combined_score = (
                phonetic_score * 0.4 +
                sequence_score * 0.3 +
                keyword_score * 0.3
            )
            
            if combined_score > 0.4:
                asr_match = ASRMatch(
                    asr_text=text,
                    canonical_verse=candidate.canonical_verse,
                    confidence_score=combined_score,
                    matching_strategy=MatchingStrategy.HYBRID,
                    match_details={
                        'phonetic_score': phonetic_score,
                        'sequence_score': sequence_score,
                        'keyword_score': keyword_score,
                        'combined_score': combined_score
                    }
                )
                matches.append(asr_match)
        
        return matches
    
    def _normalize_for_fuzzy(self, text: str) -> str:
        """Normalize text for fuzzy matching"""
        # Remove diacritics
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Lowercase and remove extra spaces
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def _calculate_keyword_score(self, asr_text: str, verse: Dict[str, Any]) -> float:
        """Calculate keyword matching score"""
        # Extract keywords from verse
        tags = verse.get('tags', [])
        keywords = set(tags)
        
        # Add important words from translation
        translation = verse.get('translation', '')
        if translation:
            important_words = ['soul', 'dharma', 'karma', 'yoga', 'eternal', 
                             'divine', 'krishna', 'arjuna', 'duty', 'action']
            for word in important_words:
                if word in translation.lower():
                    keywords.add(word)
        
        # Check how many keywords appear in ASR text
        asr_lower = asr_text.lower()
        matches = sum(1 for keyword in keywords if keyword in asr_lower)
        
        return matches / len(keywords) if keywords else 0
    
    def format_match_report(self, matches: List[ASRMatch], 
                           original_asr: str,
                           max_results: int = 3) -> str:
        """
        Format a human-readable report of matches
        
        Args:
            matches: List of ASRMatch objects
            original_asr: Original ASR text
            max_results: Maximum number of results to show
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ASR SCRIPTURE MATCHING REPORT")
        lines.append("=" * 80)
        lines.append(f"\nOriginal ASR Text:\n{original_asr}\n")
        lines.append("-" * 40)
        
        if not matches:
            lines.append("\nNo matches found with sufficient confidence.")
        else:
            lines.append(f"\nTop {min(len(matches), max_results)} Matches:\n")
            
            for i, match in enumerate(matches[:max_results], 1):
                lines.append(f"{i}. {match.verse_reference}")
                lines.append(f"   Confidence: {match.confidence_score:.2%}")
                lines.append(f"   Strategy: {match.matching_strategy.value}")
                lines.append(f"   Canonical Text: {match.canonical_text[:100]}...")
                
                if match.canonical_verse.get('translation'):
                    translation = match.canonical_verse['translation']
                    lines.append(f"   Translation: {translation[:100]}...")
                
                lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the matcher
    matcher = ASRScriptureMatcher(Path("data/scriptures"))
    
    # Test cases from Digital Dharma research
    test_cases = [
        # Abbreviated reference
        "In Gita ch 3, verse 44, Krishna explains the nature of desire",
        
        # Garbled ASR output (attempting BG 4.7)
        "yada yada he dharmasya glan ir bavat ebharata",
        
        # Another garbled example (attempting BG 2.47)
        "karmany eva dhikaras te ma phaleshu kadachana",
        
        # Mixed English and Sanskrit
        "The verse says avyakto yam achintyo yam something about unchanging",
    ]
    
    for test_text in test_cases:
        print(f"\nTesting: {test_text}")
        print("-" * 40)
        
        matches = matcher.match_asr_to_verse(test_text)
        report = matcher.format_match_report(matches, test_text)
        print(report)