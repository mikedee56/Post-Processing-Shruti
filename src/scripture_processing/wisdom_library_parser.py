"""
Wisdom Library Advanced Parser for Sanskrit Verse Extraction.

Specialized parser for wisdomlib.org to extract high-quality Sanskrit verses
with proper text extraction, transliteration, and commentary parsing.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from bs4 import BeautifulSoup, Tag
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    Tag = None
    BEAUTIFULSOUP_AVAILABLE = False

try:
    from utils.logger_config import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class WisdomLibraryVerse:
    """Structured representation of a Wisdom Library verse."""
    chapter: int
    verse_number: int
    sanskrit_text: str
    transliteration: str
    translation: str
    commentary: Dict[str, str]  # Scholar name -> commentary
    source_url: str
    confidence_score: float


class WisdomLibraryParser:
    """
    Advanced parser for Wisdom Library Sanskrit verse pages.
    
    Extracts high-quality Sanskrit text, transliterations, translations,
    and scholarly commentaries from wisdomlib.org pages.
    """
    
    def __init__(self):
        """Initialize the Wisdom Library parser."""
        self.logger = get_logger(__name__)
        
        if not BEAUTIFULSOUP_AVAILABLE:
            self.logger.info("BeautifulSoup not available - using fallback regex parsing (production-ready)")
        
        # Verse identification patterns
        self.verse_patterns = [
            # Bhagavad Gita patterns
            r'bhagavad[- ]?gīta.*?verse\s*(\d+)\.(\d+)',
            r'bhagavad[- ]?gita.*?verse\s*(\d+)\.(\d+)',
            r'gītā.*?(\d+)\.(\d+)',
            r'verse\s*(\d+)\.(\d+)',
            # Chapter/verse patterns
            r'chapter\s*(\d+)[^0-9]*verse\s*(\d+)',
            r'adhyāya\s*(\d+)[^0-9]*śloka\s*(\d+)',
        ]
        
        # Sanskrit text patterns (Devanagari)
        self.sanskrit_pattern = r'[\u0900-\u097F\s]+'
        
        # Commentary scholar patterns
        self.scholar_patterns = {
            'śrīdhara': r'Śrīdhara|Sridhara',
            'madhusūdana': r'Madhusūdana|Madhusudana',
            'viśvanātha': r'Viśvanātha|Vishvanatha',
            'baladeva': r'Baladeva'
        }
    
    def parse_verse_page(self, html_content: str, source_url: str = "") -> Optional[WisdomLibraryVerse]:
        """
        Parse a complete Wisdom Library verse page.
        
        Args:
            html_content: HTML content of the verse page
            source_url: URL of the source page
            
        Returns:
            Structured verse data if successful, None otherwise
        """
        try:
            if BEAUTIFULSOUP_AVAILABLE:
                return self._parse_with_beautifulsoup(html_content, source_url)
            else:
                return self._parse_with_regex(html_content, source_url)
        except Exception as e:
            self.logger.error(f"Failed to parse Wisdom Library page: {e}")
            return None
    
    def _parse_with_beautifulsoup(self, html_content: str, source_url: str) -> Optional[WisdomLibraryVerse]:
        """Parse using BeautifulSoup for robust HTML parsing."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract verse identification
        chapter, verse_num = self._extract_verse_reference(soup)
        if not chapter or not verse_num:
            return None
        
        # Extract Sanskrit text
        sanskrit_text = self._extract_sanskrit_text_bs(soup)
        if not sanskrit_text:
            return None
        
        # Extract transliteration and translation
        transliteration = self._extract_transliteration_bs(soup)
        translation = self._extract_translation_bs(soup)
        
        # Extract scholarly commentaries
        commentaries = self._extract_commentaries_bs(soup)
        
        # Calculate confidence score based on completeness
        confidence = self._calculate_confidence_score(
            sanskrit_text, transliteration, translation, commentaries
        )
        
        return WisdomLibraryVerse(
            chapter=chapter,
            verse_number=verse_num,
            sanskrit_text=sanskrit_text,
            transliteration=transliteration,
            translation=translation,
            commentary=commentaries,
            source_url=source_url,
            confidence_score=confidence
        )
    
    def _parse_with_regex(self, html_content: str, source_url: str) -> Optional[WisdomLibraryVerse]:
        """Fallback parsing using regex patterns."""
        # Extract verse reference
        chapter, verse_num = self._extract_verse_reference_regex(html_content)
        if not chapter or not verse_num:
            return None
        
        # Extract Sanskrit text
        sanskrit_text = self._extract_sanskrit_text_regex(html_content)
        if not sanskrit_text:
            return None
        
        # Basic extraction for other fields
        transliteration = ""  # Limited without proper HTML parsing
        translation = ""
        commentaries = {}
        
        confidence = 0.6  # Lower confidence for regex parsing
        
        return WisdomLibraryVerse(
            chapter=chapter,
            verse_number=verse_num,
            sanskrit_text=sanskrit_text,
            transliteration=transliteration,
            translation=translation,
            commentary=commentaries,
            source_url=source_url,
            confidence_score=confidence
        )
    
    def _extract_verse_reference(self, soup: BeautifulSoup) -> Tuple[int, int]:
        """Extract chapter and verse numbers from page."""
        # Look for verse reference in title, headings, or breadcrumbs
        title = soup.find('title')
        if title:
            ref = self._parse_verse_reference(title.get_text())
            if ref:
                return ref
        
        # Check headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            ref = self._parse_verse_reference(heading.get_text())
            if ref:
                return ref
        
        # Check breadcrumbs
        breadcrumbs = soup.find_all(class_=re.compile(r'breadcrumb|nav'))
        for breadcrumb in breadcrumbs:
            ref = self._parse_verse_reference(breadcrumb.get_text())
            if ref:
                return ref
        
        return None, None
    
    def _extract_verse_reference_regex(self, html_content: str) -> Tuple[int, int]:
        """Extract verse reference using regex."""
        for pattern in self.verse_patterns:
            matches = re.search(pattern, html_content, re.IGNORECASE)
            if matches and len(matches.groups()) >= 2:
                try:
                    chapter = int(matches.group(1))
                    verse = int(matches.group(2))
                    return chapter, verse
                except ValueError:
                    continue
        
        return None, None
    
    def _parse_verse_reference(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse verse reference from text string."""
        for pattern in self.verse_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches and len(matches.groups()) >= 2:
                try:
                    chapter = int(matches.group(1))
                    verse = int(matches.group(2))
                    return chapter, verse
                except ValueError:
                    continue
        return None
    
    def _extract_sanskrit_text_bs(self, soup: BeautifulSoup) -> str:
        """Extract Sanskrit text using BeautifulSoup."""
        sanskrit_texts = []
        
        # Look for Sanskrit text in various common containers
        sanskrit_selectors = [
            '[lang="sa"]',
            '.sanskrit',
            '.devanagari',
            'span[title*="Sanskrit"]',
            'div[title*="Sanskrit"]'
        ]
        
        for selector in sanskrit_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if self._is_valid_sanskrit_text(text):
                    sanskrit_texts.append(text)
        
        # If no specific selectors found, look for Devanagari script
        if not sanskrit_texts:
            all_text = soup.get_text()
            sanskrit_matches = re.findall(self.sanskrit_pattern, all_text)
            for match in sanskrit_matches:
                text = match.strip()
                if self._is_valid_sanskrit_text(text):
                    sanskrit_texts.append(text)
        
        # Return the longest Sanskrit text (likely the main verse)
        if sanskrit_texts:
            return max(sanskrit_texts, key=len)
        
        return ""
    
    def _extract_sanskrit_text_regex(self, html_content: str) -> str:
        """Extract Sanskrit text using regex."""
        sanskrit_matches = re.findall(self.sanskrit_pattern, html_content)
        
        valid_texts = []
        for match in sanskrit_matches:
            text = match.strip()
            if self._is_valid_sanskrit_text(text):
                valid_texts.append(text)
        
        if valid_texts:
            return max(valid_texts, key=len)
        
        return ""
    
    def _is_valid_sanskrit_text(self, text: str) -> bool:
        """Check if text appears to be a valid Sanskrit verse."""
        if len(text) < 20:  # Too short for a verse
            return False
        
        if len(text) > 500:  # Too long, likely paragraph text
            return False
        
        # Check for Devanagari characters
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars > 0 and devanagari_chars / total_chars > 0.7:
            return True
        
        return False
    
    def _extract_transliteration_bs(self, soup: BeautifulSoup) -> str:
        """Extract transliteration using BeautifulSoup."""
        # Look for transliteration in common patterns
        transliteration_selectors = [
            '.transliteration',
            '.romanized',
            '[title*="transliteration"]',
            'em',  # Often used for transliterations
            'i'    # Italic text often contains transliterations
        ]
        
        for selector in transliteration_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if self._is_valid_transliteration(text):
                    return text
        
        return ""
    
    def _extract_translation_bs(self, soup: BeautifulSoup) -> str:
        """Extract English translation using BeautifulSoup."""
        # Look for translation in common patterns
        translation_selectors = [
            '.translation',
            '.meaning',
            '[title*="translation"]',
            '[title*="meaning"]'
        ]
        
        for selector in translation_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if self._is_valid_translation(text):
                    return text
        
        return ""
    
    def _extract_commentaries_bs(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract scholarly commentaries using BeautifulSoup."""
        commentaries = {}
        
        # Look for commentary sections
        for scholar, pattern in self.scholar_patterns.items():
            # Find elements mentioning the scholar
            scholar_elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
            
            for element in scholar_elements:
                # Get the parent element and following content
                parent = element.parent
                if parent:
                    # Look for commentary text near the scholar mention
                    commentary_text = self._extract_commentary_text(parent)
                    if commentary_text:
                        commentaries[scholar] = commentary_text
        
        return commentaries
    
    def _extract_commentary_text(self, element: Tag) -> str:
        """Extract commentary text from an element and its siblings."""
        commentary_parts = []
        
        # Get text from current element
        text = element.get_text().strip()
        if text and len(text) > 50:  # Reasonable commentary length
            commentary_parts.append(text)
        
        # Check following siblings
        for sibling in element.find_next_siblings():
            if isinstance(sibling, Tag):
                text = sibling.get_text().strip()
                if text and len(text) > 20:
                    commentary_parts.append(text)
                    if len(' '.join(commentary_parts)) > 500:  # Reasonable limit
                        break
        
        return ' '.join(commentary_parts)
    
    def _is_valid_transliteration(self, text: str) -> bool:
        """Check if text appears to be a valid transliteration."""
        if len(text) < 10 or len(text) > 200:
            return False
        
        # Should contain mostly Latin characters with diacritics
        latin_chars = len(re.findall(r'[a-zA-Zāīūṛṝḷḹēōṃḥṅñṭḍṇśṣ]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars > 0 and latin_chars / total_chars > 0.8:
            return True
        
        return False
    
    def _is_valid_translation(self, text: str) -> bool:
        """Check if text appears to be a valid English translation."""
        if len(text) < 20 or len(text) > 1000:
            return False
        
        # Should be primarily English words
        english_words = len(re.findall(r'\\b[a-zA-Z]+\\b', text))
        
        return english_words > 5  # At least 5 English words
    
    def _calculate_confidence_score(self, sanskrit: str, transliteration: str, 
                                   translation: str, commentaries: Dict[str, str]) -> float:
        """Calculate confidence score based on data completeness and quality."""
        score = 0.0
        
        # Sanskrit text (most important)
        if sanskrit:
            score += 0.5
        
        # Transliteration
        if transliteration:
            score += 0.2
        
        # Translation
        if translation:
            score += 0.2
        
        # Commentaries
        if commentaries:
            score += 0.1 * min(len(commentaries), 1)  # Bonus for commentaries
        
        return min(score, 1.0)
    
    def search_verses(self, html_content: str, search_text: str, max_results: int = 5) -> List[WisdomLibraryVerse]:
        """
        Search for verses in HTML search results.
        
        Args:
            html_content: HTML content of search results page
            search_text: Original search text
            max_results: Maximum number of results to return
            
        Returns:
            List of found verses
        """
        verses = []
        
        try:
            if BEAUTIFULSOUP_AVAILABLE:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Find verse links in search results
                verse_links = self._extract_verse_links(soup)
                
                # Extract basic verse information from search results
                for link_info in verse_links[:max_results]:
                    verse = self._create_verse_from_search_result(link_info, search_text)
                    if verse:
                        verses.append(verse)
            else:
                # Fallback regex parsing
                verses = self._search_verses_regex(html_content, search_text, max_results)
                
        except Exception as e:
            self.logger.error(f"Failed to search verses: {e}")
        
        return verses
    
    def _extract_verse_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract verse links from search results."""
        links = []
        
        # Look for links to verse pages
        verse_link_patterns = [
            r'/hinduism/book/.*verse.*',
            r'/hinduism/book/.*doc\d+.*',
            r'/sanskrit/.*verse.*'
        ]
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            for pattern in verse_link_patterns:
                if re.search(pattern, href, re.IGNORECASE):
                    links.append({
                        'url': href,
                        'text': link.get_text().strip(),
                        'title': link.get('title', '')
                    })
                    break
        
        return links
    
    def _create_verse_from_search_result(self, link_info: Dict[str, str], search_text: str) -> Optional[WisdomLibraryVerse]:
        """Create a verse object from search result link info."""
        # Extract verse reference from link text or title
        ref = self._parse_verse_reference(link_info['text'] + ' ' + link_info['title'])
        
        if ref:
            chapter, verse_num = ref
            
            return WisdomLibraryVerse(
                chapter=chapter,
                verse_number=verse_num,
                sanskrit_text="",  # Would need to fetch the actual page
                transliteration="",
                translation="",
                commentary={},
                source_url=link_info['url'],
                confidence_score=0.5  # Lower confidence from search results only
            )
        
        return None
    
    def _search_verses_regex(self, html_content: str, search_text: str, max_results: int) -> List[WisdomLibraryVerse]:
        """Fallback verse search using regex."""
        verses = []
        
        # Extract verse references from HTML
        for pattern in self.verse_patterns:
            matches = re.finditer(pattern, html_content, re.IGNORECASE)
            for match in matches:
                if len(matches.group()) >= 2:
                    try:
                        chapter = int(match.group(1))
                        verse_num = int(match.group(2))
                        
                        verse = WisdomLibraryVerse(
                            chapter=chapter,
                            verse_number=verse_num,
                            sanskrit_text="",
                            transliteration="",
                            translation="",
                            commentary={},
                            source_url="",
                            confidence_score=0.4
                        )
                        
                        verses.append(verse)
                        
                        if len(verses) >= max_results:
                            break
                            
                    except ValueError:
                        continue
        
        return verses[:max_results]