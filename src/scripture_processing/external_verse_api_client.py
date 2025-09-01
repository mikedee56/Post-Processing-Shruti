"""
External Verse API Client for legitimate Sanskrit scripture integration.

Integrates with verified external APIs to improve verse identification accuracy
from 40% to target 70%+ using canonical scripture databases.
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from utils.logger_config import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

try:
    from utils.error_handler import ErrorHandler
except ImportError:
    class ErrorHandler:
        def __init__(self, component):
            self.logger = logging.getLogger(component)
        def log_operation_start(self, op, data): pass
        def log_operation_success(self, op, data): pass
        def log_operation_error(self, op, error, data): pass

try:
    from sanskrit_hindi_identifier.lexicon_manager import LexiconManager
except ImportError:
    # Fallback class for environments without lexicon manager
    class LexiconManager:
        def __init__(self):
            pass

try:
    from scripture_processing.wisdom_library_parser import WisdomLibraryParser
    WISDOM_PARSER_AVAILABLE = True
except ImportError:
    WisdomLibraryParser = None
    WISDOM_PARSER_AVAILABLE = False


logger = get_logger(__name__)


class APIProvider(Enum):
    """Supported external verse API providers."""
    BHAGAVAD_GITA_API = "bhagavad_gita_api"
    RAPID_API = "rapid_api"
    WISDOM_LIB = "wisdom_lib"
    WISDOM_LIB_SCRAPER = "wisdom_lib_scraper"


@dataclass
class VerseReference:
    """Standard verse reference format."""
    scripture: str  # e.g., "bhagavad_gita", "upanishads"
    chapter: int
    verse: int
    text_sanskrit: str
    text_transliteration: str
    translation: str
    source: str


@dataclass
class APIResponse:
    """Standardized API response format."""
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    provider: APIProvider
    response_time: float


class ExternalVerseAPIClient:
    """
    Client for accessing legitimate external Sanskrit verse APIs.
    
    Integrates with verified APIs to enhance verse identification accuracy
    while maintaining academic integrity and proper attribution.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the external API client."""
        self.config = config or {}
        self.error_handler = ErrorHandler(component="ExternalVerseAPIClient")
        
        # Configure HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # API configurations
        self.api_configs = {
            APIProvider.BHAGAVAD_GITA_API: {
                "base_url": "https://vedicscriptures.github.io/",
                "rate_limit": 100,  # requests per minute
                "timeout": 10
            },
            APIProvider.RAPID_API: {
                "base_url": "https://rapidapi.com/bhagavad-gita-bhagavad-gita-default/api/bhagavad-gita3",
                "headers": {
                    "X-RapidAPI-Key": self.config.get("rapid_api_key", ""),
                    "X-RapidAPI-Host": "bhagavad-gita3.p.rapidapi.com"
                },
                "rate_limit": 500,
                "timeout": 15
            },
            APIProvider.WISDOM_LIB_SCRAPER: {
                "base_url": "https://www.wisdomlib.org",
                "rate_limit": 60,  # Conservative rate limiting for web scraping
                "timeout": 20,
                "headers": {
                    "User-Agent": "Sanskrit-Research-Tool/1.0 (Academic Research)"
                }
            }
        }
        
        # Local cache for API responses
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour TTL
        
        # Initialize Wisdom Library parser if available
        if WISDOM_PARSER_AVAILABLE:
            self.wisdom_parser = WisdomLibraryParser()
            logger.info("Wisdom Library advanced parser initialized")
        else:
            self.wisdom_parser = None
            logger.warning("Wisdom Library parser not available, using basic parsing")
        
        logger.info(f"ExternalVerseAPIClient initialized with {len(self.api_configs)} providers")
    
    def search_verse_by_text(self, text_fragment: str, 
                           max_results: int = 5,
                           similarity_threshold: float = 0.7) -> List[VerseReference]:
        """
        Search for verses matching the given text fragment.
        
        Args:
            text_fragment: Text to search for
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for matches
            
        Returns:
            List of matching verse references
        """
        cache_key = hashlib.md5(f"{text_fragment}_{max_results}_{similarity_threshold}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Returning cached result for verse search")
                return cached_result
        
        results = []
        
        # Try Wisdom Library first (most comprehensive and accurate)
        try:
            wisdom_results = self._search_wisdom_library(text_fragment, max_results)
            results.extend(wisdom_results)
            logger.info(f"Wisdom Library returned {len(wisdom_results)} results")
        except Exception as e:
            logger.warning(f"Wisdom Library search failed: {e}")
        
        # Try Bhagavad Gita API as backup (still reliable)
        if len(results) < max_results:
            try:
                bg_results = self._search_bhagavad_gita_api(text_fragment, max_results - len(results))
                results.extend(bg_results)
                logger.info(f"Bhagavad Gita API returned {len(bg_results)} additional results")
            except Exception as e:
                logger.warning(f"Bhagavad Gita API search failed: {e}")
        
        # Filter by similarity threshold
        filtered_results = [r for r in results if self._calculate_similarity(text_fragment, r.text_sanskrit) >= similarity_threshold]
        
        # Cache results
        self.cache[cache_key] = (filtered_results, time.time())
        
        logger.info(f"Found {len(filtered_results)} verse matches for text fragment")
        return filtered_results[:max_results]
    
    def get_verse_by_reference(self, scripture: str, chapter: int, verse: int) -> Optional[VerseReference]:
        """
        Get a specific verse by scripture reference.
        
        Args:
            scripture: Scripture name (e.g., "bhagavad_gita")
            chapter: Chapter number
            verse: Verse number
            
        Returns:
            Verse reference if found, None otherwise
        """
        cache_key = f"{scripture}_{chapter}_{verse}"
        
        # Check cache
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        try:
            # Try Wisdom Library first (most comprehensive)
            if scripture.lower() in ["bhagavad_gita", "gita", "bhagavad-gita"]:
                result = self._get_wisdom_library_verse(scripture, chapter, verse)
                if result:
                    self.cache[cache_key] = (result, time.time())
                    return result
                
                # Fallback to GitHub API
                result = self._get_bhagavad_gita_verse(chapter, verse)
                if result:
                    self.cache[cache_key] = (result, time.time())
                    return result
        except Exception as e:
            logger.error(f"Failed to fetch verse {scripture} {chapter}.{verse}: {e}")
        
        return None
    
    def _search_bhagavad_gita_api(self, text_fragment: str, max_results: int) -> List[VerseReference]:
        """Search using the open-source Bhagavad Gita API."""
        results = []
        
        # The GitHub API provides JSON files for each chapter
        # We'll need to search through chapters to find matching text
        for chapter in range(1, 19):  # Bhagavad Gita has 18 chapters
            try:
                response = self._make_api_request(
                    APIProvider.BHAGAVAD_GITA_API,
                    f"chapter{chapter}.json"
                )
                
                if response.success and response.data:
                    chapter_verses = response.data
                    for verse_data in chapter_verses:
                        if self._text_matches(text_fragment, verse_data.get('text_sanskrit', '')):
                            verse_ref = VerseReference(
                                scripture="bhagavad_gita",
                                chapter=chapter,
                                verse=verse_data.get('verse', 0),
                                text_sanskrit=verse_data.get('text_sanskrit', ''),
                                text_transliteration=verse_data.get('transliteration', ''),
                                translation=verse_data.get('translation', ''),
                                source="vedicscriptures_api"
                            )
                            results.append(verse_ref)
                            
                            if len(results) >= max_results:
                                return results
                        
            except Exception as e:
                logger.warning(f"Error searching chapter {chapter}: {e}")
                continue
        
        return results
    
    def _search_wisdom_library(self, text_fragment: str, max_results: int) -> List[VerseReference]:
        """
        Search Wisdom Library using intelligent web scraping.
        
        This method uses the site's search functionality and structured data
        to find verses matching the text fragment.
        """
        results = []
        
        try:
            # Use Wisdom Library's search functionality
            search_url = f"{self.api_configs[APIProvider.WISDOM_LIB_SCRAPER]['base_url']}/search"
            
            # Search for the text fragment
            search_params = {
                'q': text_fragment,
                'category': 'sanskrit',
                'limit': max_results * 2  # Get more results to filter
            }
            
            response = self._make_wisdom_library_request(search_url, params=search_params)
            
            if response.success and response.data:
                # Parse search results using advanced parser if available
                if self.wisdom_parser:
                    wisdom_verses = self.wisdom_parser.search_verses(response.data, text_fragment, max_results)
                    results = [self._convert_wisdom_verse_to_reference(wv) for wv in wisdom_verses]
                else:
                    # Fallback to basic parsing
                    results = self._parse_wisdom_library_search_results(response.data, text_fragment, max_results)
                
            logger.info(f"Wisdom Library search returned {len(results)} results")
                
        except Exception as e:
            logger.error(f"Wisdom Library search failed: {e}")
        
        return results[:max_results]
    
    def _get_wisdom_library_verse(self, scripture: str, chapter: int, verse: int) -> Optional[VerseReference]:
        """
        Get a specific verse from Wisdom Library.
        
        Uses the structured URL pattern for direct verse access.
        """
        try:
            # Construct Wisdom Library URL for the specific verse
            if scripture.lower() in ["bhagavad_gita", "gita", "bhagavad-gita"]:
                # Example URL pattern: /hinduism/book/bhagavad-gita-with-four-commentaries-sanskrit/d/doc[ID].html
                verse_url = self._construct_wisdom_library_verse_url("bhagavad-gita", chapter, verse)
                
                response = self._make_wisdom_library_request(verse_url)
                
                if response.success and response.data:
                    if self.wisdom_parser:
                        wisdom_verse = self.wisdom_parser.parse_verse_page(response.data, verse_url)
                        if wisdom_verse:
                            return self._convert_wisdom_verse_to_reference(wisdom_verse)
                    else:
                        return self._parse_wisdom_library_verse_page(response.data, scripture, chapter, verse)
                    
        except Exception as e:
            logger.error(f"Failed to fetch Wisdom Library verse {scripture} {chapter}.{verse}: {e}")
        
        return None
    
    def _construct_wisdom_library_verse_url(self, scripture: str, chapter: int, verse: int) -> str:
        """
        Construct URL for specific verse on Wisdom Library.
        
        Note: This is a simplified approach. In practice, you'd need to map
        chapter/verse numbers to their specific document IDs on Wisdom Library.
        """
        base_path = "/hinduism/book/bhagavad-gita-with-four-commentaries-sanskrit/d"
        
        # For now, we'll construct a search-based approach since we don't have
        # the exact mapping of verse numbers to document IDs
        search_query = f"bhagavad gita chapter {chapter} verse {verse}"
        return f"/search?q={search_query.replace(' ', '+')}&category=sanskrit"
    
    def _make_wisdom_library_request(self, url_or_path: str, params: Dict = None) -> APIResponse:
        """
        Make a request to Wisdom Library with proper headers and rate limiting.
        """
        start_time = time.time()
        
        try:
            config = self.api_configs[APIProvider.WISDOM_LIB_SCRAPER]
            
            # Construct full URL if path is provided
            if not url_or_path.startswith('http'):
                url = f"{config['base_url']}{url_or_path}"
            else:
                url = url_or_path
            
            headers = config.get("headers", {})
            timeout = config.get("timeout", 20)
            
            # Add respectful delay for web scraping
            time.sleep(1)  # 1 second delay between requests
            
            response = self.session.get(url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            
            response_time = time.time() - start_time
            
            # For HTML responses, we'll pass the text content
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type:
                data = response.text
            else:
                try:
                    data = response.json()
                except:
                    data = response.text
            
            return APIResponse(
                success=True,
                data=data,
                error=None,
                provider=APIProvider.WISDOM_LIB_SCRAPER,
                response_time=response_time
            )
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            logger.error(f"Wisdom Library request failed: {e}")
            
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                provider=APIProvider.WISDOM_LIB_SCRAPER,
                response_time=response_time
            )
    
    def _parse_wisdom_library_search_results(self, html_content: str, 
                                           search_text: str, 
                                           max_results: int) -> List[VerseReference]:
        """
        Parse Wisdom Library search results HTML to extract verse information.
        """
        results = []
        
        try:
            # Basic HTML parsing to extract verse information
            # In a production environment, you'd want to use BeautifulSoup or similar
            import re
            
            # Look for verse patterns in the HTML
            verse_patterns = [
                r'bhagavad[- ]?gita[^>]*chapter\s*(\d+)[^>]*verse\s*(\d+)',
                r'गीता\s*अध्याय\s*(\d+)\s*श्लोक\s*(\d+)',
                r'verse\s*(\d+)\.(\d+)',
            ]
            
            for pattern in verse_patterns:
                matches = re.finditer(pattern, html_content, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        chapter = int(match.group(1))
                        verse_num = int(match.group(2))
                        
                        # Create a verse reference
                        verse_ref = VerseReference(
                            scripture="bhagavad_gita",
                            chapter=chapter,
                            verse=verse_num,
                            text_sanskrit=self._extract_sanskrit_text(html_content, match.start()),
                            text_transliteration="",  # Would need more sophisticated parsing
                            translation="",  # Would need more sophisticated parsing
                            source="wisdom_library"
                        )
                        
                        results.append(verse_ref)
                        
                        if len(results) >= max_results:
                            break
                            
                if len(results) >= max_results:
                    break
                    
        except Exception as e:
            logger.warning(f"Failed to parse Wisdom Library search results: {e}")
        
        return results[:max_results]
    
    def _parse_wisdom_library_verse_page(self, html_content: str, 
                                        scripture: str, 
                                        chapter: int, 
                                        verse: int) -> Optional[VerseReference]:
        """
        Parse a Wisdom Library verse page to extract detailed verse information.
        """
        try:
            # Extract Sanskrit text, transliteration, and translation from the page
            import re
            
            # Look for Sanskrit text patterns
            sanskrit_text = self._extract_sanskrit_from_page(html_content)
            transliteration = self._extract_transliteration_from_page(html_content)
            translation = self._extract_translation_from_page(html_content)
            
            if sanskrit_text:
                return VerseReference(
                    scripture=scripture,
                    chapter=chapter,
                    verse=verse,
                    text_sanskrit=sanskrit_text,
                    text_transliteration=transliteration or "",
                    translation=translation or "",
                    source="wisdom_library"
                )
                
        except Exception as e:
            logger.warning(f"Failed to parse Wisdom Library verse page: {e}")
        
        return None
    
    def _extract_sanskrit_text(self, html_content: str, position: int) -> str:
        """Extract Sanskrit text near a position in HTML content."""
        # Simplified extraction - in practice, would use proper HTML parsing
        import re
        
        # Look for Devanagari text patterns
        devanagari_pattern = r'[\u0900-\u097F]+'
        matches = list(re.finditer(devanagari_pattern, html_content))
        
        # Find the closest match to the position
        closest_match = None
        min_distance = float('inf')
        
        for match in matches:
            distance = abs(match.start() - position)
            if distance < min_distance:
                min_distance = distance
                closest_match = match
        
        if closest_match and len(closest_match.group()) > 10:  # Reasonable verse length
            return closest_match.group().strip()
        
        return ""
    
    def _extract_sanskrit_from_page(self, html_content: str) -> str:
        """Extract main Sanskrit verse from a Wisdom Library page."""
        import re
        
        # Look for the main Sanskrit verse (usually in a specific div or span)
        devanagari_pattern = r'[\u0900-\u097F\s]+'
        matches = re.findall(devanagari_pattern, html_content)
        
        # Return the longest Sanskrit text (likely the main verse)
        if matches:
            longest_match = max(matches, key=len)
            if len(longest_match.strip()) > 15:  # Minimum verse length
                return longest_match.strip()
        
        return ""
    
    def _extract_transliteration_from_page(self, html_content: str) -> str:
        """Extract transliteration from a Wisdom Library page."""
        # This would require sophisticated parsing of the page structure
        # For now, return empty - would need BeautifulSoup for proper implementation
        return ""
    
    def _extract_translation_from_page(self, html_content: str) -> str:
        """Extract English translation from a Wisdom Library page."""
        # This would require sophisticated parsing of the page structure
        # For now, return empty - would need BeautifulSoup for proper implementation
        return ""
    
    def _convert_wisdom_verse_to_reference(self, wisdom_verse) -> VerseReference:
        """
        Convert WisdomLibraryVerse to VerseReference format.
        
        Args:
            wisdom_verse: WisdomLibraryVerse object
            
        Returns:
            VerseReference object
        """
        return VerseReference(
            scripture="bhagavad_gita",
            chapter=wisdom_verse.chapter,
            verse=wisdom_verse.verse_number,
            text_sanskrit=wisdom_verse.sanskrit_text,
            text_transliteration=wisdom_verse.transliteration,
            translation=wisdom_verse.translation,
            source=f"wisdom_library_{wisdom_verse.confidence_score:.2f}"
        )
    
    def _get_bhagavad_gita_verse(self, chapter: int, verse: int) -> Optional[VerseReference]:
        """Get a specific Bhagavad Gita verse."""
        try:
            response = self._make_api_request(
                APIProvider.BHAGAVAD_GITA_API,
                f"chapter{chapter}.json"
            )
            
            if response.success and response.data:
                for verse_data in response.data:
                    if verse_data.get('verse') == verse:
                        return VerseReference(
                            scripture="bhagavad_gita",
                            chapter=chapter,
                            verse=verse,
                            text_sanskrit=verse_data.get('text_sanskrit', ''),
                            text_transliteration=verse_data.get('transliteration', ''),
                            translation=verse_data.get('translation', ''),
                            source="vedicscriptures_api"
                        )
        except Exception as e:
            logger.error(f"Error fetching verse BG {chapter}.{verse}: {e}")
        
        return None
    
    def _make_api_request(self, provider: APIProvider, endpoint: str) -> APIResponse:
        """Make a request to the specified API provider."""
        start_time = time.time()
        
        try:
            config = self.api_configs[provider]
            url = f"{config['base_url']}/{endpoint}"
            
            headers = config.get("headers", {})
            timeout = config.get("timeout", 10)
            
            response = self.session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            response_time = time.time() - start_time
            
            return APIResponse(
                success=True,
                data=response.json(),
                error=None,
                provider=provider,
                response_time=response_time
            )
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            logger.error(f"API request failed for {provider.value}: {e}")
            
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                provider=provider,
                response_time=response_time
            )
    
    def _text_matches(self, fragment: str, canonical_text: str) -> bool:
        """Check if text fragment matches canonical text."""
        # Simple fuzzy matching - can be enhanced with better algorithms
        return self._calculate_similarity(fragment, canonical_text) > 0.6
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        try:
            from fuzzywuzzy import fuzz
            return fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100.0
        except ImportError:
            # Fallback to simple character overlap
            set1 = set(text1.lower())
            set2 = set(text2.lower())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all configured API providers."""
        status = {}
        
        for provider in self.api_configs:
            try:
                # Test connection with provider-specific requests
                if provider == APIProvider.WISDOM_LIB_SCRAPER:
                    # Test Wisdom Library with a simple page request
                    response = self._make_wisdom_library_request("/")
                else:
                    # Test other APIs with standard request
                    response = self._make_api_request(provider, "chapter1.json")
                
                status[provider.value] = {
                    "available": response.success,
                    "response_time": response.response_time,
                    "error": response.error
                }
            except Exception as e:
                status[provider.value] = {
                    "available": False,
                    "response_time": 0,
                    "error": str(e)
                }
        
        return status


class EnhancedVerseIdentifier:
    """
    Enhanced verse identification system using external APIs.
    
    Combines local lexicon matching with external API verification
    to improve accuracy from 40% to target 70%+.
    """
    
    def __init__(self, lexicon_manager: LexiconManager, config: Dict = None):
        """Initialize enhanced verse identifier."""
        self.lexicon_manager = lexicon_manager
        self.config = config or {}
        self.api_client = ExternalVerseAPIClient(self.config.get('api_config', {}))
        self.error_handler = ErrorHandler(component="EnhancedVerseIdentifier")
        
        # Hybrid matching configuration
        self.use_external_apis = self.config.get('use_external_apis', True)
        self.fallback_to_local = self.config.get('fallback_to_local', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        logger.info("EnhancedVerseIdentifier initialized with external API support")
    
    def identify_verses(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify verses in text using hybrid local + external approach.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of identified verses with enhanced confidence scores
        """
        identified_verses = []
        
        try:
            # Step 1: Local lexicon identification (baseline)
            local_matches = self._identify_local_verses(text)
            
            # Step 2: External API verification and enhancement
            if self.use_external_apis:
                enhanced_matches = self._enhance_with_external_apis(local_matches, text)
                identified_verses.extend(enhanced_matches)
            else:
                identified_verses.extend(local_matches)
            
            # Step 3: Confidence scoring and deduplication
            identified_verses = self._deduplicate_and_score(identified_verses)
            
            logger.info(f"Identified {len(identified_verses)} verses with enhanced accuracy")
            
        except Exception as e:
            self.error_handler.log_operation_error("verse_identification", e, {"text_length": len(text)})
            if self.fallback_to_local:
                return self._identify_local_verses(text)
            raise
        
        return identified_verses
    
    def _identify_local_verses(self, text: str) -> List[Dict[str, Any]]:
        """Use local lexicon for verse identification."""
        # This would integrate with existing ScriptureIdentifier
        try:
            from scripture_processing.scripture_identifier import ScriptureIdentifier
            identifier = ScriptureIdentifier(self.lexicon_manager)
            return identifier.identify_scripture_passages(text)
        except ImportError:
            logger.warning("Local scripture identifier not available")
            return []
    
    def _enhance_with_external_apis(self, local_matches: List[Dict], text: str) -> List[Dict[str, Any]]:
        """Enhance local matches with external API verification."""
        enhanced_matches = []
        
        for match in local_matches:
            try:
                # Extract relevant text fragment
                fragment = match.get('matched_text', '')
                
                # Search external APIs for verification
                api_results = self.api_client.search_verse_by_text(
                    fragment,
                    max_results=3,
                    similarity_threshold=0.6
                )
                
                if api_results:
                    # Enhance match with external verification
                    best_api_match = max(api_results, 
                                       key=lambda x: self._calculate_combined_confidence(match, x))
                    
                    enhanced_match = {
                        **match,
                        'external_verification': True,
                        'canonical_text': best_api_match.text_sanskrit,
                        'transliteration': best_api_match.text_transliteration,
                        'translation': best_api_match.translation,
                        'scripture_reference': f"{best_api_match.scripture}_{best_api_match.chapter}.{best_api_match.verse}",
                        'enhanced_confidence': self._calculate_combined_confidence(match, best_api_match),
                        'source': best_api_match.source
                    }
                    enhanced_matches.append(enhanced_match)
                else:
                    # Keep original match but mark as unverified
                    match['external_verification'] = False
                    match['enhanced_confidence'] = match.get('confidence_score', 0.5) * 0.8  # Reduce confidence
                    enhanced_matches.append(match)
                    
            except Exception as e:
                logger.warning(f"External API enhancement failed for match: {e}")
                # Fallback to original match
                match['external_verification'] = False
                enhanced_matches.append(match)
        
        return enhanced_matches
    
    def _calculate_combined_confidence(self, local_match: Dict, api_match: VerseReference) -> float:
        """Calculate combined confidence score from local and external matches."""
        local_confidence = local_match.get('confidence_score', 0.5)
        
        # Calculate text similarity
        text_similarity = self.api_client._calculate_similarity(
            local_match.get('matched_text', ''),
            api_match.text_sanskrit
        )
        
        # Combined scoring: 60% local + 40% external verification
        combined_confidence = (local_confidence * 0.6) + (text_similarity * 0.4)
        
        return min(combined_confidence, 1.0)
    
    def _deduplicate_and_score(self, verses: List[Dict]) -> List[Dict[str, Any]]:
        """Remove duplicates and finalize confidence scores."""
        # Simple deduplication by scripture reference
        seen_refs = set()
        unique_verses = []
        
        for verse in verses:
            ref = verse.get('scripture_reference', verse.get('matched_text', ''))
            if ref not in seen_refs:
                seen_refs.add(ref)
                unique_verses.append(verse)
        
        # Sort by confidence score
        unique_verses.sort(key=lambda x: x.get('enhanced_confidence', x.get('confidence_score', 0)), reverse=True)
        
        return unique_verses
    
    def validate_verse_accuracy(self, identified_verses: List[Dict], 
                              golden_references: List[Dict]) -> Dict[str, float]:
        """
        Validate verse identification accuracy against golden dataset.
        
        Args:
            identified_verses: Verses identified by the system
            golden_references: Known correct verse references
            
        Returns:
            Accuracy metrics
        """
        if not golden_references:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
        
        # Calculate precision and recall
        identified_refs = {v.get('scripture_reference', '') for v in identified_verses}
        golden_refs = {g.get('scripture_reference', '') for g in golden_references}
        
        true_positives = len(identified_refs & golden_refs)
        false_positives = len(identified_refs - golden_refs)
        false_negatives = len(golden_refs - identified_refs)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        accuracy = true_positives / len(golden_refs) if golden_refs else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }