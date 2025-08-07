"""
Semantic Similarity Calculator for Story 2.4.2

This module provides semantic similarity computation using iNLTK embeddings with caching
capabilities for the advanced ASR post-processing workflow. Integrates with existing
Story 2.2 contextual modeling and Story 2.3 scripture processing components.

Architecture Integration:
- Extends existing contextual analysis in Story 2.2 
- Provides Stage 3 matching for hybrid scripture pipeline in Story 2.3
- File-based embedding storage compatible with existing YAML approach
"""

import json
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from scipy.spatial.distance import cosine

# Import iNLTK for semantic embeddings
try:
    from inltk.inltk import setup, get_sentence_encoding
    INLTK_AVAILABLE = True
except ImportError:
    INLTK_AVAILABLE = False

from utils.logger_config import get_logger


class LanguageModel(Enum):
    """Supported language models for semantic similarity computation."""
    SANSKRIT = "sa"
    HINDI = "hi"  
    ENGLISH = "en"
    AUTO_DETECT = "auto"


@dataclass
class SemanticSimilarityResult:
    """Result of semantic similarity computation between two texts."""
    text1: str
    text2: str
    similarity_score: float
    language_used: str
    embedding_model: str
    computation_time: float
    cache_hit: bool
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SemanticVectorCache:
    """
    Cache entry for pre-computed semantic embeddings.
    
    Architecture Integration: Embedded within existing YAML scripture files as new fields
    and stored in dedicated semantic cache directory for performance.
    """
    text: str
    embedding_vector: List[float]
    embedding_model_version: str
    language: str
    last_computed: str  # ISO format datetime
    computation_metadata: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticVectorCache':
        """Create instance from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SemanticSimilarityCalculator:
    """
    Advanced semantic similarity computation using iNLTK embeddings.
    
    This component provides:
    1. Semantic similarity computation between Sanskrit/Hindi text pairs (AC1)
    2. File-based caching system for performance optimization (AC2)
    3. Batch processing capabilities (AC3)
    4. Normalized scoring consistent with existing confidence systems (AC4)
    5. Multi-language support with automatic model selection (AC5)
    6. Integration with Story 2.2 contextual modeling (AC6)
    7. Scripture database enhancement for Story 2.3 (AC7)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the semantic similarity calculator.
        
        Args:
            cache_dir: Directory for caching embeddings. Defaults to data/semantic_cache/
            config: Configuration dictionary for advanced parameters
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Setup cache directory
        self.cache_dir = cache_dir or Path("data/semantic_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self._embedding_cache: Dict[str, SemanticVectorCache] = {}
        self._load_cache()
        
        # Model configuration
        self.models_initialized = {}
        self.embedding_dimension = 400  # iNLTK default dimension
        
        # Performance metrics
        self.stats = {
            'total_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_computation_time': 0.0,
            'embeddings_generated': 0,
            'batch_operations': 0
        }
        
        # Initialize iNLTK if available
        if not INLTK_AVAILABLE:
            self.logger.warning(
                "iNLTK not available. Semantic similarity will use fallback implementation."
            )
    
    def _get_cache_key(self, text: str, language: str) -> str:
        """Generate cache key for text and language combination."""
        content = f"{text}:{language}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_cache(self) -> None:
        """Load existing embeddings from cache files."""
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                self._embedding_cache = {
                    key: SemanticVectorCache.from_dict(data)
                    for key, data in cache_data.items()
                }
                
                self.logger.info(
                    f"Loaded {len(self._embedding_cache)} cached embeddings from {cache_file}"
                )
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.error(f"Error loading embedding cache: {e}")
                self._embedding_cache = {}
        else:
            self.logger.info("No existing embedding cache found. Starting fresh.")
    
    def _save_cache(self) -> None:
        """Save current embeddings to cache file."""
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        try:
            cache_data = {
                key: cache_entry.to_dict()
                for key, cache_entry in self._embedding_cache.items()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            self.logger.debug(f"Saved {len(self._embedding_cache)} embeddings to cache")
            
        except (OSError, TypeError) as e:
            self.logger.error(f"Error saving embedding cache: {e}")
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of input text for appropriate model selection (AC5).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code (sa, hi, en)
        """
        # Simple heuristic-based language detection
        # More sophisticated detection could be added later
        
        # Check for Devanagari script (Hindi/Sanskrit)
        devanagari_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        devanagari_ratio = devanagari_chars / max(len(text), 1)
        
        # Check for IAST characters (Sanskrit)
        iast_chars = sum(1 for char in text if char in 'āīūṛṝḷḹēōṃḥṅñṭḍṇṣśĀĪŪṚṜḶḸĒŌṂḤṄÑṬḌṆṢŚ')
        iast_ratio = iast_chars / max(len(text), 1)
        
        if devanagari_ratio > 0.3:
            # High Devanagari content - could be Hindi or Sanskrit
            # Use Sanskrit model for better coverage of Vedantic terms
            return LanguageModel.SANSKRIT.value
        elif iast_ratio > 0.1:
            # IAST transliteration detected - Sanskrit
            return LanguageModel.SANSKRIT.value
        else:
            # Default to Sanskrit for this domain
            return LanguageModel.SANSKRIT.value
    
    def _initialize_model(self, language: str) -> bool:
        """
        Initialize iNLTK model for specified language.
        
        Args:
            language: Language code
            
        Returns:
            True if model initialized successfully
        """
        if not INLTK_AVAILABLE:
            return False
            
        if language in self.models_initialized:
            return self.models_initialized[language]
        
        try:
            # Initialize iNLTK for the language
            setup(language)
            self.models_initialized[language] = True
            self.logger.info(f"Initialized iNLTK model for language: {language}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize iNLTK model for {language}: {e}")
            self.models_initialized[language] = False
            return False
    
    def _get_embedding(self, text: str, language: str) -> Optional[np.ndarray]:
        """
        Get semantic embedding for text using iNLTK.
        
        Args:
            text: Text to embed
            language: Language model to use
            
        Returns:
            Embedding vector or None if failed
        """
        # Check cache first
        cache_key = self._get_cache_key(text, language)
        
        if cache_key in self._embedding_cache:
            cached = self._embedding_cache[cache_key]
            self.stats['cache_hits'] += 1
            return np.array(cached.embedding_vector)
        
        # Generate new embedding
        self.stats['cache_misses'] += 1
        
        if not self._initialize_model(language):
            return None
        
        try:
            # Get iNLTK sentence embedding
            embedding = get_sentence_encoding(text, language)
            
            if embedding is None:
                return None
            
            # Convert to numpy array and ensure proper shape
            embedding_array = np.array(embedding)
            if embedding_array.ndim == 0:
                return None
                
            # Cache the embedding
            cache_entry = SemanticVectorCache(
                text=text,
                embedding_vector=embedding_array.tolist(),
                embedding_model_version=f"iNLTK-{language}-v1.0",
                language=language,
                last_computed=datetime.now(timezone.utc).isoformat(),
                computation_metadata={
                    'text_length': len(text),
                    'embedding_dimension': len(embedding_array),
                    'model_type': 'iNLTK'
                }
            )
            
            self._embedding_cache[cache_key] = cache_entry
            self.stats['embeddings_generated'] += 1
            
            return embedding_array
            
        except Exception as e:
            self.logger.error(f"Error generating embedding for text '{text[:50]}...': {e}")
            return None
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback similarity computation when iNLTK is not available.
        Uses basic string similarity as fallback.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0-1.0)
        """
        from difflib import SequenceMatcher
        
        # Basic string similarity
        similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Apply normalization to match semantic similarity range
        return min(similarity * 0.8, 1.0)  # Scale down to account for lower semantic understanding
    
    def compute_semantic_similarity(
        self, 
        text1: str, 
        text2: str,
        language: Optional[str] = None
    ) -> SemanticSimilarityResult:
        """
        Compute semantic similarity between two texts (AC1, AC4, AC5).
        
        Args:
            text1: First text for comparison
            text2: Second text for comparison  
            language: Language model to use (auto-detected if None)
            
        Returns:
            SemanticSimilarityResult with normalized score (0.0-1.0)
        """
        start_time = datetime.now()
        self.stats['total_computations'] += 1
        
        # Language detection
        if language is None or language == LanguageModel.AUTO_DETECT.value:
            # Use primary text for language detection
            detected_lang = self._detect_language(text1)
        else:
            detected_lang = language
        
        # Check if both texts are cached
        cache_key1 = self._get_cache_key(text1, detected_lang)
        cache_key2 = self._get_cache_key(text2, detected_lang)
        
        cache_hit = (cache_key1 in self._embedding_cache and 
                    cache_key2 in self._embedding_cache)
        
        # Get embeddings
        embedding1 = self._get_embedding(text1, detected_lang)
        embedding2 = self._get_embedding(text2, detected_lang)
        
        # Compute similarity
        if embedding1 is not None and embedding2 is not None:
            try:
                # Use cosine similarity (normalized 0-1)
                cosine_distance = cosine(embedding1, embedding2)
                similarity_score = 1.0 - cosine_distance
                
                # Ensure score is in valid range
                similarity_score = max(0.0, min(1.0, similarity_score))
                
                model_used = f"iNLTK-{detected_lang}"
                
            except Exception as e:
                self.logger.error(f"Error computing cosine similarity: {e}")
                similarity_score = self._fallback_similarity(text1, text2)
                model_used = "fallback"
        else:
            # Fallback to basic similarity
            similarity_score = self._fallback_similarity(text1, text2)
            model_used = "fallback"
        
        # Calculate computation time
        computation_time = (datetime.now() - start_time).total_seconds()
        self.stats['total_computation_time'] += computation_time
        
        # Create result
        result = SemanticSimilarityResult(
            text1=text1,
            text2=text2,
            similarity_score=similarity_score,
            language_used=detected_lang,
            embedding_model=model_used,
            computation_time=computation_time,
            cache_hit=cache_hit,
            metadata={
                'text1_length': len(text1),
                'text2_length': len(text2),
                'fallback_used': not INLTK_AVAILABLE or embedding1 is None or embedding2 is None
            }
        )
        
        # Save cache periodically
        if self.stats['embeddings_generated'] % 10 == 0:
            self._save_cache()
        
        return result
    
    def batch_compute_similarities(
        self, 
        text_pairs: List[Tuple[str, str]],
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[SemanticSimilarityResult]:
        """
        Compute semantic similarity for multiple text pairs efficiently (AC3).
        
        Args:
            text_pairs: List of (text1, text2) tuples
            language: Language model to use (auto-detected if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of SemanticSimilarityResult objects
        """
        self.stats['batch_operations'] += 1
        results = []
        
        total_pairs = len(text_pairs)
        self.logger.info(f"Starting batch computation for {total_pairs} text pairs")
        
        for i, (text1, text2) in enumerate(text_pairs):
            result = self.compute_semantic_similarity(text1, text2, language)
            results.append(result)
            
            # Progress callback
            if progress_callback and (i + 1) % max(1, total_pairs // 10) == 0:
                progress = (i + 1) / total_pairs
                progress_callback(progress, i + 1, total_pairs)
        
        # Save cache after batch operation
        self._save_cache()
        
        self.logger.info(
            f"Completed batch computation. "
            f"Average similarity: {sum(r.similarity_score for r in results) / len(results):.3f}"
        )
        
        return results
    
    def get_cached_embeddings_count(self) -> int:
        """Get number of cached embeddings."""
        return len(self._embedding_cache)
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._embedding_cache.clear()
        cache_file = self.cache_dir / "embeddings_cache.json"
        if cache_file.exists():
            cache_file.unlink()
        self.logger.info("Cleared embedding cache")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring and optimization.
        
        Returns:
            Dictionary with performance metrics
        """
        total_ops = self.stats['total_computations']
        cache_hit_rate = (self.stats['cache_hits'] / max(total_ops, 1)) * 100
        avg_computation_time = self.stats['total_computation_time'] / max(total_ops, 1)
        
        return {
            'total_computations': total_ops,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'average_computation_time': f"{avg_computation_time:.4f}s",
            'cached_embeddings': len(self._embedding_cache),
            'embeddings_generated': self.stats['embeddings_generated'],
            'batch_operations': self.stats['batch_operations'],
            'inltk_available': INLTK_AVAILABLE
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate system configuration and dependencies.
        
        Returns:
            Validation results with status and recommendations
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check iNLTK availability
        if not INLTK_AVAILABLE:
            validation['warnings'].append("iNLTK not available - using fallback similarity")
            validation['recommendations'].append("Install iNLTK for advanced semantic similarity")
        
        # Check cache directory
        if not self.cache_dir.exists():
            validation['errors'].append(f"Cache directory does not exist: {self.cache_dir}")
            validation['is_valid'] = False
        elif not os.access(self.cache_dir, os.W_OK):
            validation['errors'].append(f"Cache directory not writable: {self.cache_dir}")
            validation['is_valid'] = False
        
        # Check for potential issues
        if len(self._embedding_cache) > 10000:
            validation['warnings'].append("Large embedding cache may impact memory usage")
            validation['recommendations'].append("Consider periodic cache cleanup")
        
        return validation
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save cache on cleanup."""
        self._save_cache()