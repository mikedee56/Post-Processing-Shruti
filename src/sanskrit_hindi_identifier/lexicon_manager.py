"""
Enhanced Lexicon Management System.

This module provides comprehensive lexicon management capabilities including
loading, validation, updating, and version control of Sanskrit/Hindi lexicons.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
import hashlib

from utils.logger_config import get_logger
from sanskrit_hindi_identifier.word_identifier import LexiconEntry, WordCategory


class LexiconFormat(Enum):
    """Supported lexicon file formats."""
    YAML = "yaml"
    JSON = "json"


@dataclass
class LexiconMetadata:
    """Metadata for lexicon files."""
    name: str
    version: str
    format: LexiconFormat
    last_updated: str
    entries_count: int
    checksum: str
    source_authority: str
    description: str
    categories: List[str]


@dataclass
class LexiconValidationResult:
    """Result of lexicon validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]


class LexiconManager:
    """
    Enhanced lexicon management system.
    
    Provides centralized management of all lexicon files with validation,
    version control, and dynamic loading capabilities.
    """

    def __init__(self, lexicon_dir: Path = None, enable_caching: bool = True):
        """
        Initialize the lexicon manager.
        
        Args:
            lexicon_dir: Directory containing lexicon files
            enable_caching: Whether to enable in-memory caching
        """
        self.logger = get_logger(__name__)
        self.lexicon_dir = lexicon_dir or Path("data/lexicons")
        self.enable_caching = enable_caching
        
        # Initialize data structures
        self.lexicons: Dict[str, Dict[str, LexiconEntry]] = {}
        self.metadata: Dict[str, LexiconMetadata] = {}
        self.cached_data: Dict[str, Dict] = {} if enable_caching else None
        
        # Supported file extensions
        self.supported_extensions = {'.yaml', '.yml', '.json'}
        
        # Initialize the manager
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the lexicon manager."""
        if not self.lexicon_dir.exists():
            self.lexicon_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"Created lexicon directory: {self.lexicon_dir}")
        
        # Load all available lexicons
        self.load_all_lexicons()
        
        self.logger.info(f"LexiconManager initialized with {len(self.lexicons)} lexicons")

    def load_all_lexicons(self) -> Dict[str, LexiconValidationResult]:
        """
        Load all lexicon files from the lexicon directory.
        
        Returns:
            Dictionary mapping file names to validation results
        """
        results = {}
        
        for file_path in self.lexicon_dir.iterdir():
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                try:
                    result = self.load_lexicon(file_path)
                    results[file_path.name] = result
                except Exception as e:
                    self.logger.error(f"Failed to load lexicon {file_path}: {e}")
                    results[file_path.name] = LexiconValidationResult(
                        is_valid=False,
                        errors=[str(e)],
                        warnings=[],
                        statistics={}
                    )
        
        return results

    def load_lexicon(self, file_path: Path) -> LexiconValidationResult:
        """
        Load a single lexicon file.
        
        Args:
            file_path: Path to the lexicon file
            
        Returns:
            LexiconValidationResult with loading and validation results
        """
        file_name = file_path.name
        
        try:
            # Determine format
            if file_path.suffix in ['.yaml', '.yml']:
                format_type = LexiconFormat.YAML
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            elif file_path.suffix == '.json':
                format_type = LexiconFormat.JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Validate structure
            validation_result = self.validate_lexicon_data(data, file_name)
            
            if validation_result.is_valid:
                # Parse entries
                entries = self._parse_lexicon_entries(data.get('entries', []), file_name)
                
                # Store in memory
                self.lexicons[file_name] = entries
                
                # Generate metadata
                metadata = self._generate_metadata(file_path, format_type, data, len(entries))
                self.metadata[file_name] = metadata
                
                # Cache if enabled
                if self.enable_caching:
                    self.cached_data[file_name] = data
                
                self.logger.info(f"Loaded lexicon {file_name}: {len(entries)} entries")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error loading lexicon {file_path}: {e}")
            return LexiconValidationResult(
                is_valid=False,
                errors=[str(e)],
                warnings=[],
                statistics={}
            )

    def validate_lexicon_data(self, data: Dict, file_name: str) -> LexiconValidationResult:
        """
        Validate lexicon data structure and content.
        
        Args:
            data: Lexicon data to validate
            file_name: Name of the file being validated
            
        Returns:
            LexiconValidationResult with validation results
        """
        errors = []
        warnings = []
        statistics = {}
        
        # Check basic structure
        if not isinstance(data, dict):
            errors.append("Lexicon must be a dictionary")
            return LexiconValidationResult(False, errors, warnings, statistics)
        
        if 'entries' not in data:
            errors.append("Lexicon must contain 'entries' key")
            return LexiconValidationResult(False, errors, warnings, statistics)
        
        entries = data['entries']
        if not isinstance(entries, list):
            errors.append("'entries' must be a list")
            return LexiconValidationResult(False, errors, warnings, statistics)
        
        # Validate individual entries
        valid_entries = 0
        categories = set()
        authorities = set()
        duplicates = set()
        seen_terms = set()
        
        for i, entry in enumerate(entries):
            entry_errors = self._validate_entry(entry, i)
            if entry_errors:
                errors.extend(entry_errors)
            else:
                valid_entries += 1
                
                # Collect statistics
                if 'category' in entry:
                    categories.add(entry['category'])
                if 'source_authority' in entry:
                    authorities.add(entry['source_authority'])
                
                # Check for duplicates
                term = entry.get('original_term', '').lower()
                if term in seen_terms:
                    duplicates.add(term)
                    warnings.append(f"Duplicate term found: {term}")
                else:
                    seen_terms.add(term)
        
        # Compile statistics
        statistics = {
            'total_entries': len(entries),
            'valid_entries': valid_entries,
            'invalid_entries': len(entries) - valid_entries,
            'categories': list(categories),
            'source_authorities': list(authorities),
            'duplicates_found': len(duplicates),
            'unique_terms': len(seen_terms)
        }
        
        # Add warnings for potential issues
        if len(duplicates) > 0:
            warnings.append(f"Found {len(duplicates)} duplicate terms")
        
        if valid_entries < len(entries) * 0.9:  # Less than 90% valid
            warnings.append("High number of invalid entries detected")
        
        is_valid = len(errors) == 0
        
        return LexiconValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )

    def _validate_entry(self, entry: Dict, index: int) -> List[str]:
        """Validate a single lexicon entry."""
        errors = []
        
        if not isinstance(entry, dict):
            errors.append(f"Entry {index}: must be a dictionary")
            return errors
        
        # Required fields
        required_fields = ['original_term', 'transliteration']
        for field in required_fields:
            if field not in entry:
                errors.append(f"Entry {index}: missing required field '{field}'")
            elif not entry[field] or not isinstance(entry[field], str):
                errors.append(f"Entry {index}: '{field}' must be a non-empty string")
        
        # Optional but recommended fields
        recommended_fields = ['variations', 'is_proper_noun', 'category', 'confidence']
        for field in recommended_fields:
            if field not in entry:
                # Only warning level, not error
                pass
        
        # Validate specific field types and values
        if 'variations' in entry:
            if not isinstance(entry['variations'], list):
                errors.append(f"Entry {index}: 'variations' must be a list")
        
        if 'is_proper_noun' in entry:
            if not isinstance(entry['is_proper_noun'], bool):
                errors.append(f"Entry {index}: 'is_proper_noun' must be a boolean")
        
        if 'confidence' in entry:
            confidence = entry['confidence']
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                errors.append(f"Entry {index}: 'confidence' must be a number between 0 and 1")
        
        if 'category' in entry:
            category = entry['category']
            if not isinstance(category, str):
                errors.append(f"Entry {index}: 'category' must be a string")
            # Check if category is valid (optional validation)
            valid_categories = {cat.value for cat in WordCategory}
            if category not in valid_categories:
                # This is just a warning, not an error
                pass
        
        return errors

    def _parse_lexicon_entries(self, entries_data: List[Dict], source_file: str) -> Dict[str, LexiconEntry]:
        """Parse raw entry data into LexiconEntry objects."""
        entries = {}
        
        for entry_data in entries_data:
            try:
                entry = LexiconEntry(
                    original_term=entry_data.get('original_term', ''),
                    variations=entry_data.get('variations', []),
                    transliteration=entry_data.get('transliteration', ''),
                    is_proper_noun=entry_data.get('is_proper_noun', False),
                    category=entry_data.get('category', 'unknown'),
                    confidence=entry_data.get('confidence', 1.0),
                    source_authority=entry_data.get('source_authority', source_file)
                )
                
                entries[entry.original_term.lower()] = entry
                
            except Exception as e:
                self.logger.error(f"Error parsing entry from {source_file}: {e}")
        
        return entries

    def _generate_metadata(self, file_path: Path, format_type: LexiconFormat, 
                          data: Dict, entries_count: int) -> LexiconMetadata:
        """Generate metadata for a lexicon file."""
        # Calculate checksum
        content = file_path.read_text(encoding='utf-8')
        checksum = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Extract categories
        categories = []
        if 'entries' in data:
            categories = list(set(
                entry.get('category', 'unknown') 
                for entry in data['entries']
                if isinstance(entry, dict)
            ))
        
        return LexiconMetadata(
            name=file_path.stem,
            version=data.get('version', '1.0'),
            format=format_type,
            last_updated=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            entries_count=entries_count,
            checksum=checksum,
            source_authority=data.get('source_authority', 'unknown'),
            description=data.get('description', ''),
            categories=categories
        )

    def get_all_entries(self) -> Dict[str, LexiconEntry]:
        """Get all entries from all loaded lexicons."""
        all_entries = {}
        
        for lexicon_name, entries in self.lexicons.items():
            for term, entry in entries.items():
                # Handle conflicts by keeping the one with higher confidence
                if term in all_entries:
                    if entry.confidence > all_entries[term].confidence:
                        all_entries[term] = entry
                else:
                    all_entries[term] = entry
        
        return all_entries

    def get_entries_by_category(self, category: Union[str, WordCategory]) -> Dict[str, LexiconEntry]:
        """Get all entries for a specific category."""
        if isinstance(category, WordCategory):
            category = category.value
        
        entries = {}
        for lexicon_entries in self.lexicons.values():
            for term, entry in lexicon_entries.items():
                if entry.category == category:
                    entries[term] = entry
        
        return entries

    def search_entries(self, query: str, max_results: int = 10) -> List[LexiconEntry]:
        """Search for entries matching a query."""
        results = []
        query_lower = query.lower()
        
        all_entries = self.get_all_entries()
        
        # Exact matches first
        for term, entry in all_entries.items():
            if query_lower == term:
                results.append(entry)
        
        # Partial matches
        for term, entry in all_entries.items():
            if query_lower in term and entry not in results:
                results.append(entry)
        
        # Variation matches
        for term, entry in all_entries.items():
            if any(query_lower in var.lower() for var in entry.variations) and entry not in results:
                results.append(entry)
        
        return results[:max_results]

    def add_entry(self, lexicon_name: str, entry: LexiconEntry) -> bool:
        """
        Add a new entry to a lexicon.
        
        Args:
            lexicon_name: Name of the lexicon file
            entry: LexiconEntry to add
            
        Returns:
            True if successfully added, False otherwise
        """
        if lexicon_name not in self.lexicons:
            self.logger.error(f"Lexicon {lexicon_name} not found")
            return False
        
        term_key = entry.original_term.lower()
        
        # Check for duplicates
        if term_key in self.lexicons[lexicon_name]:
            self.logger.warning(f"Entry {entry.original_term} already exists in {lexicon_name}")
            return False
        
        # Add the entry
        self.lexicons[lexicon_name][term_key] = entry
        
        # Update metadata
        if lexicon_name in self.metadata:
            self.metadata[lexicon_name].entries_count += 1
            self.metadata[lexicon_name].last_updated = datetime.now().isoformat()
        
        self.logger.info(f"Added entry {entry.original_term} to {lexicon_name}")
        return True

    def remove_entry(self, lexicon_name: str, term: str) -> bool:
        """
        Remove an entry from a lexicon.
        
        Args:
            lexicon_name: Name of the lexicon file
            term: Term to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        if lexicon_name not in self.lexicons:
            self.logger.error(f"Lexicon {lexicon_name} not found")
            return False
        
        term_key = term.lower()
        
        if term_key not in self.lexicons[lexicon_name]:
            self.logger.warning(f"Entry {term} not found in {lexicon_name}")
            return False
        
        # Remove the entry
        del self.lexicons[lexicon_name][term_key]
        
        # Update metadata
        if lexicon_name in self.metadata:
            self.metadata[lexicon_name].entries_count -= 1
            self.metadata[lexicon_name].last_updated = datetime.now().isoformat()
        
        self.logger.info(f"Removed entry {term} from {lexicon_name}")
        return True

    def save_lexicon(self, lexicon_name: str, backup: bool = True) -> bool:
        """
        Save a lexicon back to file.
        
        Args:
            lexicon_name: Name of the lexicon to save
            backup: Whether to create a backup first
            
        Returns:
            True if successfully saved, False otherwise
        """
        if lexicon_name not in self.lexicons:
            self.logger.error(f"Lexicon {lexicon_name} not found")
            return False
        
        file_path = self.lexicon_dir / lexicon_name
        
        try:
            # Create backup if requested
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}{file_path.suffix}')
                backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
                self.logger.info(f"Created backup: {backup_path}")
            
            # Prepare data for saving
            entries_data = []
            for entry in self.lexicons[lexicon_name].values():
                entries_data.append(asdict(entry))
            
            save_data = {
                'version': self.metadata[lexicon_name].version if lexicon_name in self.metadata else '1.0',
                'description': self.metadata[lexicon_name].description if lexicon_name in self.metadata else '',
                'source_authority': self.metadata[lexicon_name].source_authority if lexicon_name in self.metadata else 'lexicon_manager',
                'last_updated': datetime.now().isoformat(),
                'entries': entries_data
            }
            
            # Save based on format
            if file_path.suffix in ['.yaml', '.yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(save_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            elif file_path.suffix == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved lexicon {lexicon_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving lexicon {lexicon_name}: {e}")
            return False

    def get_lexicon_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all loaded lexicons."""
        stats = {
            'total_lexicons': len(self.lexicons),
            'total_entries': 0,
            'lexicon_details': {},
            'category_distribution': {},
            'authority_distribution': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        for lexicon_name, entries in self.lexicons.items():
            lexicon_stats = {
                'entries_count': len(entries),
                'categories': set(),
                'authorities': set(),
                'avg_confidence': 0,
                'proper_nouns': 0
            }
            
            total_confidence = 0
            for entry in entries.values():
                stats['total_entries'] += 1
                
                # Category distribution
                category = entry.category
                lexicon_stats['categories'].add(category)
                stats['category_distribution'][category] = stats['category_distribution'].get(category, 0) + 1
                
                # Authority distribution
                authority = entry.source_authority
                lexicon_stats['authorities'].add(authority)
                stats['authority_distribution'][authority] = stats['authority_distribution'].get(authority, 0) + 1
                
                # Confidence distribution
                if entry.confidence >= 0.9:
                    stats['confidence_distribution']['high'] += 1
                elif entry.confidence >= 0.7:
                    stats['confidence_distribution']['medium'] += 1
                else:
                    stats['confidence_distribution']['low'] += 1
                
                total_confidence += entry.confidence
                
                if entry.is_proper_noun:
                    lexicon_stats['proper_nouns'] += 1
            
            lexicon_stats['avg_confidence'] = total_confidence / len(entries) if entries else 0
            lexicon_stats['categories'] = list(lexicon_stats['categories'])
            lexicon_stats['authorities'] = list(lexicon_stats['authorities'])
            
            stats['lexicon_details'][lexicon_name] = lexicon_stats
        
        return stats

    def get_metadata(self, lexicon_name: str = None) -> Union[LexiconMetadata, Dict[str, LexiconMetadata]]:
        """Get metadata for one or all lexicons."""
        if lexicon_name:
            return self.metadata.get(lexicon_name)
        return self.metadata.copy()

    def refresh_lexicons(self) -> Dict[str, LexiconValidationResult]:
        """Refresh all lexicons by reloading from files."""
        self.lexicons.clear()
        self.metadata.clear()
        if self.cached_data:
            self.cached_data.clear()
        
        return self.load_all_lexicons()