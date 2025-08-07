"""
Semantic Cache Manager for Story 2.4.2

This module provides advanced caching management for semantic embeddings,
including cache validation, cleanup, migration utilities, and integration
with scripture database enhancement.

Architecture Integration:
- Supports enhanced YAML scripture schema with semantic_embedding fields
- Provides migration utilities for existing scripture files
- Optimizes file-based embedding storage for performance
"""

import json
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import logging

import yaml
from dataclasses import dataclass

from utils.logger_config import get_logger
from .semantic_similarity_calculator import SemanticVectorCache


@dataclass
class CacheStatistics:
    """Statistics for cache performance monitoring."""
    total_entries: int
    cache_size_mb: float
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    languages: Set[str]
    model_versions: Set[str]
    invalid_entries: int


class SemanticCacheManager:
    """
    Advanced cache management for semantic embeddings.
    
    This component provides:
    1. Cache validation and cleanup utilities
    2. Migration support for existing scripture files (AC7)
    3. Batch embedding computation for scripture database
    4. Cache optimization and maintenance
    5. Integration with Story 2.3 scripture YAML schema
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the semantic cache manager.
        
        Args:
            cache_dir: Directory for caching embeddings
        """
        self.logger = get_logger(__name__)
        self.cache_dir = cache_dir or Path("data/semantic_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Scripture integration paths
        self.scripture_dir = Path("data/scriptures")
        self.backup_dir = self.cache_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_cache(self) -> CacheStatistics:
        """
        Validate cache entries and generate statistics.
        
        Returns:
            CacheStatistics with validation results
        """
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        if not cache_file.exists():
            return CacheStatistics(
                total_entries=0,
                cache_size_mb=0.0,
                oldest_entry=None,
                newest_entry=None,
                languages=set(),
                model_versions=set(),
                invalid_entries=0
            )
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Calculate statistics
            total_entries = len(cache_data)
            cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
            languages = set()
            model_versions = set()
            dates = []
            invalid_entries = 0
            
            for key, entry_data in cache_data.items():
                try:
                    # Validate entry structure
                    if not all(field in entry_data for field in 
                              ['text', 'embedding_vector', 'language', 'last_computed']):
                        invalid_entries += 1
                        continue
                    
                    # Parse date
                    date_str = entry_data['last_computed']
                    entry_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    dates.append(entry_date)
                    
                    # Collect metadata
                    languages.add(entry_data['language'])
                    if 'embedding_model_version' in entry_data:
                        model_versions.add(entry_data['embedding_model_version'])
                    
                except (ValueError, KeyError, TypeError) as e:
                    self.logger.warning(f"Invalid cache entry {key}: {e}")
                    invalid_entries += 1
            
            # Calculate date statistics
            oldest_entry = min(dates) if dates else None
            newest_entry = max(dates) if dates else None
            
            return CacheStatistics(
                total_entries=total_entries,
                cache_size_mb=cache_size_mb,
                oldest_entry=oldest_entry,
                newest_entry=newest_entry,
                languages=languages,
                model_versions=model_versions,
                invalid_entries=invalid_entries
            )
            
        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Error validating cache: {e}")
            return CacheStatistics(
                total_entries=0,
                cache_size_mb=0.0,
                oldest_entry=None,
                newest_entry=None,
                languages=set(),
                model_versions=set(),
                invalid_entries=1  # The cache file itself is invalid
            )
    
    def cleanup_expired_entries(self, max_age_days: int = 30) -> int:
        """
        Clean up expired cache entries.
        
        Args:
            max_age_days: Maximum age for cache entries in days
            
        Returns:
            Number of entries removed
        """
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        if not cache_file.exists():
            return 0
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            entries_to_remove = []
            
            for key, entry_data in cache_data.items():
                try:
                    entry_date = datetime.fromisoformat(
                        entry_data['last_computed'].replace('Z', '+00:00')
                    )
                    
                    if entry_date < cutoff_date:
                        entries_to_remove.append(key)
                        
                except (ValueError, KeyError):
                    # Invalid entries should be removed
                    entries_to_remove.append(key)
            
            # Remove expired entries
            for key in entries_to_remove:
                del cache_data[key]
            
            # Save updated cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            removed_count = len(entries_to_remove)
            self.logger.info(f"Cleaned up {removed_count} expired cache entries")
            return removed_count
            
        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    def create_cache_backup(self) -> Path:
        """
        Create a backup of the current cache.
        
        Returns:
            Path to the backup file
        """
        cache_file = self.cache_dir / "embeddings_cache.json"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"embeddings_cache_backup_{timestamp}.json"
        
        try:
            if cache_file.exists():
                shutil.copy2(cache_file, backup_file)
                self.logger.info(f"Created cache backup: {backup_file}")
            else:
                # Create empty backup for consistency
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                self.logger.info(f"Created empty cache backup: {backup_file}")
            
            return backup_file
            
        except OSError as e:
            self.logger.error(f"Error creating cache backup: {e}")
            raise
    
    def restore_cache_backup(self, backup_file: Path) -> bool:
        """
        Restore cache from backup file.
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            True if restore successful
        """
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        try:
            if not backup_file.exists():
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
            
            # Validate backup file
            with open(backup_file, 'r', encoding='utf-8') as f:
                json.load(f)  # Just validate JSON format
            
            # Create current backup before restore
            if cache_file.exists():
                current_backup = self.backup_dir / f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                shutil.copy2(cache_file, current_backup)
                self.logger.info(f"Created pre-restore backup: {current_backup}")
            
            # Restore from backup
            shutil.copy2(backup_file, cache_file)
            self.logger.info(f"Restored cache from backup: {backup_file}")
            return True
            
        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Error restoring cache backup: {e}")
            return False
    
    def migrate_scripture_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Migrate existing scripture YAML files to include semantic embedding fields.
        
        Args:
            dry_run: If True, only analyze changes without modifying files
            
        Returns:
            Migration results and statistics
        """
        migration_results = {
            'files_processed': 0,
            'files_modified': 0,
            'verses_enhanced': 0,
            'errors': [],
            'warnings': [],
            'dry_run': dry_run
        }
        
        if not self.scripture_dir.exists():
            migration_results['warnings'].append(f"Scripture directory not found: {self.scripture_dir}")
            return migration_results
        
        # Find all YAML scripture files
        scripture_files = list(self.scripture_dir.glob("**/*.yaml")) + list(self.scripture_dir.glob("**/*.yml"))
        
        for scripture_file in scripture_files:
            try:
                migration_results['files_processed'] += 1
                
                with open(scripture_file, 'r', encoding='utf-8') as f:
                    scripture_data = yaml.safe_load(f)
                
                if not scripture_data:
                    continue
                
                # Process scripture structure
                file_modified = False
                verses_in_file = 0
                
                # Navigate scripture structure (e.g., bhagavad_gita -> chapter_X -> verse_Y)
                for source_name, source_data in scripture_data.items():
                    if not isinstance(source_data, dict):
                        continue
                    
                    for chapter_key, chapter_data in source_data.items():
                        if not isinstance(chapter_data, dict):
                            continue
                        
                        for verse_key, verse_data in chapter_data.items():
                            if not isinstance(verse_data, dict):
                                continue
                            
                            # Check if verse has canonical_text and needs semantic enhancement
                            if 'canonical_text' in verse_data:
                                verses_in_file += 1
                                
                                # Add semantic embedding placeholder if not exists
                                if 'semantic_embedding' not in verse_data:
                                    if not dry_run:
                                        verse_data['semantic_embedding'] = {
                                            'vector': None,  # To be computed later
                                            'model_version': None,
                                            'last_computed': None,
                                            'needs_computation': True
                                        }
                                    file_modified = True
                                
                                # Add source provenance if not exists
                                if 'source_provenance' not in verse_data:
                                    if not dry_run:
                                        # Default to Gold for existing scripture files
                                        verse_data['source_provenance'] = "Gold"
                                    file_modified = True
                
                # Save modified file
                if file_modified and not dry_run:
                    # Create backup first
                    backup_file = scripture_file.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
                    shutil.copy2(scripture_file, backup_file)
                    
                    # Save enhanced file
                    with open(scripture_file, 'w', encoding='utf-8') as f:
                        yaml.dump(scripture_data, f, default_flow_style=False, 
                                allow_unicode=True, sort_keys=False, indent=2)
                    
                    migration_results['files_modified'] += 1
                    self.logger.info(f"Enhanced scripture file: {scripture_file}")
                
                if file_modified:
                    migration_results['verses_enhanced'] += verses_in_file
                
            except (yaml.YAMLError, OSError) as e:
                error_msg = f"Error processing {scripture_file}: {e}"
                migration_results['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        self.logger.info(
            f"Scripture migration {'analysis' if dry_run else 'completed'}: "
            f"{migration_results['files_processed']} files processed, "
            f"{migration_results['verses_enhanced']} verses enhanced"
        )
        
        return migration_results
    
    def compute_scripture_embeddings(
        self, 
        similarity_calculator,
        language: str = "sa",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Compute semantic embeddings for all scripture verses.
        
        Args:
            similarity_calculator: SemanticSimilarityCalculator instance
            language: Language model to use
            progress_callback: Optional progress callback
            
        Returns:
            Computation results and statistics
        """
        computation_results = {
            'files_processed': 0,
            'verses_processed': 0,
            'embeddings_computed': 0,
            'embeddings_cached': 0,
            'errors': [],
            'total_computation_time': 0.0
        }
        
        start_time = datetime.now()
        
        if not self.scripture_dir.exists():
            computation_results['errors'].append(f"Scripture directory not found: {self.scripture_dir}")
            return computation_results
        
        # Find all enhanced scripture files
        scripture_files = list(self.scripture_dir.glob("**/*.yaml")) + list(self.scripture_dir.glob("**/*.yml"))
        total_files = len(scripture_files)
        
        for file_idx, scripture_file in enumerate(scripture_files):
            try:
                computation_results['files_processed'] += 1
                
                with open(scripture_file, 'r', encoding='utf-8') as f:
                    scripture_data = yaml.safe_load(f)
                
                if not scripture_data:
                    continue
                
                file_modified = False
                
                # Process verses
                for source_name, source_data in scripture_data.items():
                    if not isinstance(source_data, dict):
                        continue
                    
                    for chapter_key, chapter_data in source_data.items():
                        if not isinstance(chapter_data, dict):
                            continue
                        
                        for verse_key, verse_data in chapter_data.items():
                            if not isinstance(verse_data, dict):
                                continue
                            
                            computation_results['verses_processed'] += 1
                            
                            # Check if verse needs embedding computation
                            if ('canonical_text' in verse_data and 
                                'semantic_embedding' in verse_data and
                                verse_data['semantic_embedding'].get('needs_computation')):
                                
                                canonical_text = verse_data['canonical_text']
                                
                                # Get embedding from calculator (uses cache internally)
                                embedding = similarity_calculator._get_embedding(canonical_text, language)
                                
                                if embedding is not None:
                                    # Update verse data
                                    verse_data['semantic_embedding'] = {
                                        'vector': embedding.tolist(),
                                        'model_version': f"iNLTK-{language}-v1.0",
                                        'last_computed': datetime.now(timezone.utc).isoformat(),
                                        'needs_computation': False
                                    }
                                    
                                    computation_results['embeddings_computed'] += 1
                                    file_modified = True
                                else:
                                    computation_results['errors'].append(
                                        f"Failed to compute embedding for {scripture_file}:{verse_key}"
                                    )
                
                # Save updated file if modified
                if file_modified:
                    with open(scripture_file, 'w', encoding='utf-8') as f:
                        yaml.dump(scripture_data, f, default_flow_style=False, 
                                allow_unicode=True, sort_keys=False, indent=2)
                
                # Progress callback
                if progress_callback:
                    progress = (file_idx + 1) / total_files
                    progress_callback(progress, file_idx + 1, total_files)
                
            except (yaml.YAMLError, OSError) as e:
                error_msg = f"Error processing {scripture_file}: {e}"
                computation_results['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        # Final statistics
        end_time = datetime.now()
        computation_results['total_computation_time'] = (end_time - start_time).total_seconds()
        computation_results['embeddings_cached'] = similarity_calculator.get_cached_embeddings_count()
        
        self.logger.info(
            f"Scripture embedding computation completed: "
            f"{computation_results['embeddings_computed']} embeddings computed in "
            f"{computation_results['total_computation_time']:.2f}s"
        )
        
        return computation_results
    
    def optimize_cache_storage(self) -> Dict[str, Any]:
        """
        Optimize cache storage by removing duplicates and compacting data.
        
        Returns:
            Optimization results
        """
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        if not cache_file.exists():
            return {'optimization_skipped': 'No cache file found'}
        
        try:
            # Load cache
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            original_size = len(cache_data)
            original_file_size = cache_file.stat().st_size
            
            # Remove duplicates by text content (different keys, same text)
            seen_texts = {}
            duplicates_removed = 0
            
            keys_to_remove = []
            for key, entry in cache_data.items():
                text = entry.get('text', '')
                language = entry.get('language', '')
                text_key = f"{text}:{language}"
                
                if text_key in seen_texts:
                    # Keep the newer entry
                    existing_key = seen_texts[text_key]
                    existing_date = cache_data[existing_key].get('last_computed', '')
                    current_date = entry.get('last_computed', '')
                    
                    if current_date > existing_date:
                        keys_to_remove.append(existing_key)
                        seen_texts[text_key] = key
                    else:
                        keys_to_remove.append(key)
                    
                    duplicates_removed += 1
                else:
                    seen_texts[text_key] = key
            
            # Remove duplicates
            for key in keys_to_remove:
                del cache_data[key]
            
            # Save optimized cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            final_size = len(cache_data)
            final_file_size = cache_file.stat().st_size
            
            results = {
                'original_entries': original_size,
                'final_entries': final_size,
                'duplicates_removed': duplicates_removed,
                'original_file_size_mb': original_file_size / (1024 * 1024),
                'final_file_size_mb': final_file_size / (1024 * 1024),
                'space_saved_mb': (original_file_size - final_file_size) / (1024 * 1024)
            }
            
            self.logger.info(
                f"Cache optimization completed: removed {duplicates_removed} duplicates, "
                f"saved {results['space_saved_mb']:.2f}MB"
            )
            
            return results
            
        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Error optimizing cache: {e}")
            return {'optimization_failed': str(e)}