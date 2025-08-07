"""
Semantic Scripture Enhancer for Story 2.4.2

This module provides semantic similarity enhancement for scripture processing,
integrating with existing Story 2.3 components to add semantic embeddings and 
improve verse candidate selection using semantic matching capabilities.

Architecture Integration:
- Enhances scripture YAML schema with semantic_embedding fields (AC7)
- Provides migration utilities for existing scripture files
- Adds semantic similarity to verse candidate selection
- Maintains backward compatibility with existing Story 2.3 functionality (AC8)
"""

import json
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

from utils.logger_config import get_logger
from .canonical_text_manager import CanonicalTextManager, CanonicalVerse, ScriptureSource
from .verse_selection_system import VerseSelectionSystem, VerseCandidateScore, SelectionConfidence
from contextual_modeling.semantic_similarity_calculator import SemanticSimilarityCalculator


@dataclass
class SemanticVerseMatch:
    """Enhanced verse match with semantic similarity scores."""
    canonical_verse: CanonicalVerse
    original_text: str
    semantic_similarity: float
    traditional_confidence: float
    combined_confidence: float
    embedding_available: bool
    matching_method: str
    processing_metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Handle enum serialization
        result['canonical_verse']['source'] = self.canonical_verse.source.value
        return result


@dataclass
class SemanticEnhancementResult:
    """Result of semantic enhancement operation."""
    original_candidates: int
    semantically_enhanced: int
    new_matches_found: int
    processing_time: float
    semantic_matches: List[SemanticVerseMatch]
    enhancement_statistics: Dict[str, Any]


class SemanticScriptureEnhancer:
    """
    Semantic enhancement for scripture processing components.
    
    This component provides:
    1. Enhanced scripture YAML schema with semantic_embedding fields (AC7)
    2. Semantic similarity integration for verse candidate selection
    3. Migration utilities for existing scripture files
    4. Backward compatibility with existing Story 2.3 functionality (AC8)
    5. Performance optimization through semantic embedding caching
    """
    
    def __init__(
        self,
        semantic_calculator: SemanticSimilarityCalculator,
        canonical_manager: CanonicalTextManager,
        semantic_weight: float = 0.4,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the semantic scripture enhancer.
        
        Args:
            semantic_calculator: SemanticSimilarityCalculator instance
            canonical_manager: CanonicalTextManager instance from Story 2.3
            semantic_weight: Weight for semantic similarity in combined scoring
            config: Additional configuration options
        """
        self.logger = get_logger(__name__)
        self.semantic_calculator = semantic_calculator
        self.canonical_manager = canonical_manager
        self.semantic_weight = semantic_weight
        self.traditional_weight = 1.0 - semantic_weight
        self.config = config or {}
        
        # Performance settings
        self.max_semantic_candidates = self.config.get('max_semantic_candidates', 20)
        self.semantic_threshold = self.config.get('semantic_threshold', 0.6)
        self.enable_embedding_caching = self.config.get('enable_embedding_caching', True)
        
        # Statistics
        self.stats = {
            'verses_enhanced': 0,
            'semantic_matches_found': 0,
            'embeddings_generated': 0,
            'cache_hits': 0,
            'processing_time_total': 0.0
        }
        
        self.logger.info(
            f"Initialized SemanticScriptureEnhancer with semantic_weight: {semantic_weight}, "
            f"threshold: {self.semantic_threshold}"
        )
    
    def enhance_verse_selection(
        self,
        verse_selection_system: VerseSelectionSystem,
        input_text: str,
        max_candidates: int = 10
    ) -> List[SemanticVerseMatch]:
        """
        Enhance verse selection with semantic similarity scoring.
        
        Args:
            verse_selection_system: VerseSelectionSystem instance from Story 2.3
            input_text: Input text to match against verses
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of SemanticVerseMatch objects with enhanced scoring
        """
        start_time = datetime.now()
        
        # Get traditional verse candidates (backward compatibility)
        selection_result = verse_selection_system.select_best_verse(input_text)
        traditional_candidates = selection_result.all_candidates
        
        semantic_matches = []
        
        # Process each traditional candidate with semantic enhancement
        for candidate in traditional_candidates[:self.max_semantic_candidates]:
            try:
                # Get canonical verse details
                canonical_verse = self._get_canonical_verse_from_candidate(candidate)
                
                if not canonical_verse:
                    continue
                
                # Compute semantic similarity
                similarity_result = self.semantic_calculator.compute_semantic_similarity(
                    input_text, canonical_verse.canonical_text
                )
                
                # Create enhanced match
                semantic_match = SemanticVerseMatch(
                    canonical_verse=canonical_verse,
                    original_text=input_text,
                    semantic_similarity=similarity_result.similarity_score,
                    traditional_confidence=candidate.match_confidence,
                    combined_confidence=self._calculate_combined_confidence(
                        candidate.match_confidence, similarity_result.similarity_score
                    ),
                    embedding_available=not similarity_result.metadata.get('fallback_used', True),
                    matching_method=f"hybrid_{similarity_result.embedding_model}",
                    processing_metadata={
                        'traditional_method': candidate.matching_method,
                        'semantic_computation_time': similarity_result.computation_time,
                        'cache_hit': similarity_result.cache_hit,
                        'language_detected': similarity_result.language_used
                    }
                )
                
                semantic_matches.append(semantic_match)
                
                # Update statistics
                if similarity_result.cache_hit:
                    self.stats['cache_hits'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing candidate {candidate.verse_id}: {e}")
                continue
        
        # Sort by combined confidence
        semantic_matches.sort(key=lambda x: x.combined_confidence, reverse=True)
        
        # Limit results
        final_matches = semantic_matches[:max_candidates]
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats['verses_enhanced'] += len(final_matches)
        self.stats['semantic_matches_found'] += len([m for m in final_matches if m.semantic_similarity >= self.semantic_threshold])
        self.stats['processing_time_total'] += processing_time
        
        self.logger.info(
            f"Enhanced {len(final_matches)} verse candidates with semantic similarity. "
            f"Processing time: {processing_time:.3f}s"
        )
        
        return final_matches
    
    def _get_canonical_verse_from_candidate(self, candidate: VerseCandidateScore) -> Optional[CanonicalVerse]:
        """Get canonical verse details from verse candidate."""
        try:
            # Extract verse details from candidate ID
            verse_parts = candidate.verse_id.split('_')
            if len(verse_parts) >= 3:
                source_name = verse_parts[0]
                chapter = int(verse_parts[1])  
                verse = int(verse_parts[2])
                
                # Map to ScriptureSource enum
                source_map = {
                    'bg': ScriptureSource.BHAGAVAD_GITA,
                    'bhagavad': ScriptureSource.BHAGAVAD_GITA,
                    'gita': ScriptureSource.BHAGAVAD_GITA,
                    'upanishad': ScriptureSource.UPANISHADS,
                    'yoga': ScriptureSource.YOGA_SUTRAS,
                    'veda': ScriptureSource.VEDAS,
                    'purana': ScriptureSource.PURANAS
                }
                
                source = source_map.get(source_name.lower(), ScriptureSource.BHAGAVAD_GITA)
                
                # Create canonical verse (simplified - in practice would load from database)
                return CanonicalVerse(
                    id=candidate.verse_id,
                    source=source,
                    chapter=chapter,
                    verse=verse,
                    canonical_text=candidate.canonical_text,
                    transliteration=candidate.transliteration or "",
                    translation=None,
                    commentary=None,
                    source_authority=candidate.source_authority,
                    tags=[],
                    variations=[]
                )
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing candidate ID {candidate.verse_id}: {e}")
        
        return None
    
    def _calculate_combined_confidence(self, traditional_confidence: float, semantic_similarity: float) -> float:
        """Calculate combined confidence score from traditional and semantic measures."""
        # Weighted combination
        combined = (self.traditional_weight * traditional_confidence + 
                   self.semantic_weight * semantic_similarity)
        
        # Apply semantic threshold bonus/penalty
        if semantic_similarity >= self.semantic_threshold:
            # Bonus for high semantic similarity
            combined = min(1.0, combined * 1.1)
        else:
            # Penalty for low semantic similarity
            combined = combined * 0.9
        
        return max(0.0, min(1.0, combined))
    
    def enhance_scripture_database(
        self, 
        scripture_dir: Path,
        language: str = "sa",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Enhance existing scripture YAML files with semantic embeddings.
        
        Args:
            scripture_dir: Directory containing scripture YAML files
            language: Language model to use for embeddings
            dry_run: If True, only analyze what would be done
            
        Returns:
            Enhancement results and statistics
        """
        enhancement_results = {
            'files_processed': 0,
            'verses_processed': 0,
            'embeddings_added': 0,
            'files_backed_up': 0,
            'processing_time': 0.0,
            'errors': [],
            'dry_run': dry_run
        }
        
        start_time = datetime.now()
        
        if not scripture_dir.exists():
            enhancement_results['errors'].append(f"Scripture directory not found: {scripture_dir}")
            return enhancement_results
        
        # Find all YAML files
        yaml_files = list(scripture_dir.glob("**/*.yaml")) + list(scripture_dir.glob("**/*.yml"))
        
        for yaml_file in yaml_files:
            try:
                enhancement_results['files_processed'] += 1
                
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    scripture_data = yaml.safe_load(f)
                
                if not scripture_data:
                    continue
                
                file_modified = False
                verses_in_file = 0
                
                # Process scripture structure
                for source_name, source_data in scripture_data.items():
                    if not isinstance(source_data, dict):
                        continue
                    
                    for chapter_key, chapter_data in source_data.items():
                        if not isinstance(chapter_data, dict):
                            continue
                        
                        for verse_key, verse_data in chapter_data.items():
                            if not isinstance(verse_data, dict):
                                continue
                            
                            verses_in_file += 1
                            enhancement_results['verses_processed'] += 1
                            
                            # Check if verse needs semantic enhancement
                            canonical_text = verse_data.get('canonical_text')
                            if canonical_text and not verse_data.get('semantic_embedding'):
                                
                                if not dry_run:
                                    # Generate semantic embedding
                                    embedding_array = self.semantic_calculator._get_embedding(
                                        canonical_text, language
                                    )
                                    
                                    if embedding_array is not None:
                                        # Add semantic embedding to verse data
                                        verse_data['semantic_embedding'] = {
                                            'vector': embedding_array.tolist(),
                                            'model_version': f"iNLTK-{language}-v1.0",
                                            'last_computed': datetime.now(timezone.utc).isoformat(),
                                            'embedding_dimension': len(embedding_array),
                                            'text_hash': hash(canonical_text) % (10**8)  # For consistency checking
                                        }
                                        
                                        # Add source provenance if missing
                                        if 'source_provenance' not in verse_data:
                                            verse_data['source_provenance'] = "Gold"  # Default for existing files
                                        
                                        enhancement_results['embeddings_added'] += 1
                                        file_modified = True
                
                # Save enhanced file
                if file_modified and not dry_run:
                    # Create backup
                    backup_file = yaml_file.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
                    yaml_file.rename(backup_file)
                    enhancement_results['files_backed_up'] += 1
                    
                    # Save enhanced version
                    with open(yaml_file, 'w', encoding='utf-8') as f:
                        yaml.dump(scripture_data, f, default_flow_style=False,
                                allow_unicode=True, sort_keys=False, indent=2)
                    
                    self.logger.info(f"Enhanced scripture file: {yaml_file}")
                
            except Exception as e:
                error_msg = f"Error processing {yaml_file}: {e}"
                enhancement_results['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        # Final statistics
        end_time = datetime.now()
        enhancement_results['processing_time'] = (end_time - start_time).total_seconds()
        
        self.logger.info(
            f"Scripture database enhancement {'analysis' if dry_run else 'completed'}: "
            f"{enhancement_results['embeddings_added']} embeddings added in "
            f"{enhancement_results['processing_time']:.2f}s"
        )
        
        return enhancement_results
    
    def find_semantic_verse_matches(
        self,
        query_text: str,
        max_candidates: int = 5,
        min_similarity: float = 0.5
    ) -> List[SemanticVerseMatch]:
        """
        Find verse matches using semantic similarity search.
        
        Args:
            query_text: Text to search for semantically similar verses
            max_candidates: Maximum number of matches to return
            min_similarity: Minimum semantic similarity threshold
            
        Returns:
            List of semantically similar verse matches
        """
        # Get all available verses from canonical manager
        all_candidates = self.canonical_manager.get_verse_candidates(
            query_text, max_candidates=self.max_semantic_candidates
        )
        
        semantic_matches = []
        
        for candidate in all_candidates:
            try:
                # Compute semantic similarity
                similarity_result = self.semantic_calculator.compute_semantic_similarity(
                    query_text, candidate.canonical_text
                )
                
                # Filter by minimum similarity
                if similarity_result.similarity_score >= min_similarity:
                    canonical_verse = self._get_canonical_verse_from_candidate(candidate)
                    
                    if canonical_verse:
                        semantic_match = SemanticVerseMatch(
                            canonical_verse=canonical_verse,
                            original_text=query_text,
                            semantic_similarity=similarity_result.similarity_score,
                            traditional_confidence=0.0,  # Pure semantic search
                            combined_confidence=similarity_result.similarity_score,
                            embedding_available=not similarity_result.metadata.get('fallback_used', True),
                            matching_method="semantic_only",
                            processing_metadata={
                                'computation_time': similarity_result.computation_time,
                                'cache_hit': similarity_result.cache_hit
                            }
                        )
                        
                        semantic_matches.append(semantic_match)
                
            except Exception as e:
                self.logger.error(f"Error in semantic search for candidate {candidate.verse_id}: {e}")
                continue
        
        # Sort by semantic similarity
        semantic_matches.sort(key=lambda x: x.semantic_similarity, reverse=True)
        
        return semantic_matches[:max_candidates]
    
    def validate_semantic_consistency(
        self,
        verse_candidates: List[VerseCandidateScore]
    ) -> Dict[str, Any]:
        """
        Validate semantic consistency across verse candidates.
        
        Args:
            verse_candidates: List of verse candidates to validate
            
        Returns:
            Validation results with consistency metrics
        """
        if len(verse_candidates) < 2:
            return {
                'consistency_score': 1.0,
                'candidates_analyzed': len(verse_candidates),
                'consistency_level': 'HIGH'
            }
        
        # Extract canonical texts
        texts = [candidate.canonical_text for candidate in verse_candidates if candidate.canonical_text]
        
        if len(texts) < 2:
            return {
                'consistency_score': 1.0,
                'candidates_analyzed': len(verse_candidates),
                'consistency_level': 'HIGH'
            }
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                result = self.semantic_calculator.compute_semantic_similarity(texts[i], texts[j])
                similarities.append(result.similarity_score)
        
        # Calculate consistency metrics
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        min_similarity = min(similarities) if similarities else 0.0
        
        # Determine consistency level
        if avg_similarity >= 0.7:
            consistency_level = 'HIGH'
        elif avg_similarity >= 0.5:
            consistency_level = 'MEDIUM'  
        else:
            consistency_level = 'LOW'
        
        return {
            'consistency_score': avg_similarity,
            'candidates_analyzed': len(verse_candidates),
            'pairwise_comparisons': len(similarities),
            'average_similarity': avg_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max(similarities) if similarities else 0.0,
            'consistency_level': consistency_level,
            'all_similarities': similarities
        }
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enhancement statistics."""
        return {
            'total_verses_enhanced': self.stats['verses_enhanced'],
            'semantic_matches_found': self.stats['semantic_matches_found'],
            'embeddings_generated': self.stats['embeddings_generated'],
            'cache_hit_count': self.stats['cache_hits'],
            'total_processing_time': f"{self.stats['processing_time_total']:.2f}s",
            'semantic_weight': self.semantic_weight,
            'semantic_threshold': self.semantic_threshold,
            'max_semantic_candidates': self.max_semantic_candidates,
            'calculator_stats': self.semantic_calculator.get_performance_stats()
        }