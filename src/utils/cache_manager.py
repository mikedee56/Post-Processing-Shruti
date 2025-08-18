# Cache Manager for Story 5.1 Performance Optimization
# Intelligent caching system to eliminate the 305% variance issue

import time
import hashlib
import pickle
import threading
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from pathlib import Path
import logging
from functools import wraps, lru_cache
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    hit_count: int = 0

class LRUCacheWithStats:
    """LRU Cache with performance statistics."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                self.hits += 1
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_accessed = time.time()
                entry.hit_count += 1
                
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return entry.value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing entry
                self.cache[key].value = value
                self.cache[key].last_accessed = current_time
            else:
                # Check if we need to evict
                if len(self.cache) >= self.maxsize:
                    # Remove least recently used item
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                
                # Add new entry
                self.cache[key] = CacheEntry(
                    value=value,
                    created_at=current_time,
                    last_accessed=current_time
                )
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': hit_rate,
                'cache_size': len(self.cache),
                'max_size': self.maxsize
            }

class CacheManager:
    """
    Comprehensive cache manager for Story 5.1 performance optimization.
    
    This manager provides multiple caching strategies to eliminate the
    305% performance variance caused by repeated expensive operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.caches: Dict[str, LRUCacheWithStats] = {}
        self.persistent_cache_dir = Path(self.config.get('cache_dir', 'cache'))
        self.persistent_cache_dir.mkdir(exist_ok=True)
        
        # Initialize specialized caches
        self._init_specialized_caches()
        
        logger.info("CacheManager initialized with specialized caches")
    
    def _init_specialized_caches(self):
        """Initialize specialized caches for different components."""
        cache_configs = {
            'text_normalization': {'maxsize': 1000},
            'lexicon_lookups': {'maxsize': 500},
            'ner_processing': {'maxsize': 300},
            'sanskrit_parser': {'maxsize': 100},  # Critical for variance reduction
            'word2vec_models': {'maxsize': 10},   # Heavy models cache
            'fuzzy_matching': {'maxsize': 800},
            'iast_transliteration': {'maxsize': 400}
        }
        
        for cache_name, config in cache_configs.items():
            self.caches[cache_name] = LRUCacheWithStats(config['maxsize'])
    
    def get_cache(self, cache_name: str) -> LRUCacheWithStats:
        """Get or create a named cache."""
        if cache_name not in self.caches:
            self.caches[cache_name] = LRUCacheWithStats()
        return self.caches[cache_name]
    
    def cached_call(self, cache_name: str, key: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with caching.
        
        Args:
            cache_name: Name of the cache to use
            key: Cache key
            func: Function to execute if not cached
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Cached or computed result
        """
        cache = self.get_cache(cache_name)
        
        # Try to get from cache first
        result = cache.get(key)
        if result is not None:
            return result
        
        # Compute and cache result
        result = func(*args, **kwargs)
        cache.put(key, result)
        return result
    
    def cache_function(self, cache_name: str, key_func: Optional[Callable] = None):
        """
        Decorator for caching function results.
        
        Args:
            cache_name: Name of the cache to use
            key_func: Optional function to generate cache key
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_data = (func.__name__, args, tuple(sorted(kwargs.items())))
                    key = hashlib.md5(str(key_data).encode()).hexdigest()
                
                return self.cached_call(cache_name, key, func, *args, **kwargs)
            return wrapper
        return decorator
    
    def save_persistent_cache(self, cache_name: str, data: Any, key: str):
        """Save data to persistent cache file."""
        cache_file = self.persistent_cache_dir / f"{cache_name}_{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved persistent cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save persistent cache {cache_file}: {e}")
    
    def load_persistent_cache(self, cache_name: str, key: str) -> Optional[Any]:
        """Load data from persistent cache file."""
        cache_file = self.persistent_cache_dir / f"{cache_name}_{key}.pkl"
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Loaded persistent cache: {cache_file}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load persistent cache {cache_file}: {e}")
        return None
    
    def clear_all_caches(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        logger.info("All caches cleared")
    
    def get_cache_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        stats = {}
        total_hits = 0
        total_misses = 0
        
        for cache_name, cache in self.caches.items():
            cache_stats = cache.get_stats()
            stats[cache_name] = cache_stats
            total_hits += cache_stats['hits']
            total_misses += cache_stats['misses']
        
        # Overall statistics
        total_requests = total_hits + total_misses
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        stats['overall'] = {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate_percent': overall_hit_rate,
            'active_caches': len(self.caches)
        }
        
        return stats

# Specialized caching decorators for Story 5.1 optimization

class SanskritParserCache:
    """
    Specialized cache for sanskrit_parser operations.
    
    This cache specifically addresses the major bottleneck causing
    305% variance by caching Word2Vec model loading and parsing results.
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.model_cache = {}  # In-memory model cache
        self.parsing_cache = self.cache_manager.get_cache('sanskrit_parser')
        
    def cache_word2vec_model(self, model_path: str, model_instance: Any):
        """Cache Word2Vec model instance to prevent repeated loading."""
        self.model_cache[model_path] = model_instance
        logger.debug(f"Cached Word2Vec model: {model_path}")
    
    def get_cached_model(self, model_path: str) -> Optional[Any]:
        """Get cached Word2Vec model instance."""
        return self.model_cache.get(model_path)
    
    def cache_parsing_result(self, text: str, result: Any):
        """Cache Sanskrit parsing result."""
        key = hashlib.md5(text.encode()).hexdigest()
        self.parsing_cache.put(key, result)
    
    def get_cached_parsing_result(self, text: str) -> Optional[Any]:
        """Get cached Sanskrit parsing result."""
        key = hashlib.md5(text.encode()).hexdigest()
        return self.parsing_cache.get(key)

class TextNormalizationCache:
    """Specialized cache for text normalization operations."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.cache = self.cache_manager.get_cache('text_normalization')
    
    def cache_normalization_result(self, text: str, context: str, result: Any):
        """Cache text normalization result with context."""
        key = hashlib.md5(f"{text}:{context}".encode()).hexdigest()
        self.cache.put(key, result)
    
    def get_cached_normalization(self, text: str, context: str) -> Optional[Any]:
        """Get cached text normalization result."""
        key = hashlib.md5(f"{text}:{context}".encode()).hexdigest()
        return self.cache.get(key)

class LexiconCache:
    """Specialized cache for lexicon lookup operations."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.cache = self.cache_manager.get_cache('lexicon_lookups')
        self.fuzzy_cache = self.cache_manager.get_cache('fuzzy_matching')
    
    def cache_lexicon_lookup(self, term: str, result: Any):
        """Cache lexicon lookup result."""
        key = term.lower()
        self.cache.put(key, result)
    
    def get_cached_lexicon_lookup(self, term: str) -> Optional[Any]:
        """Get cached lexicon lookup result."""
        key = term.lower()
        return self.cache.get(key)
    
    def cache_fuzzy_match(self, term: str, candidates: tuple, result: Any):
        """Cache fuzzy matching result."""
        key = hashlib.md5(f"{term}:{candidates}".encode()).hexdigest()
        self.fuzzy_cache.put(key, result)
    
    def get_cached_fuzzy_match(self, term: str, candidates: tuple) -> Optional[Any]:
        """Get cached fuzzy matching result."""
        key = hashlib.md5(f"{term}:{candidates}".encode()).hexdigest()
        return self.fuzzy_cache.get(key)

# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None

def get_global_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager

def reset_global_cache():
    """Reset global cache manager."""
    global _global_cache_manager
    if _global_cache_manager:
        _global_cache_manager.clear_all_caches()
    _global_cache_manager = None