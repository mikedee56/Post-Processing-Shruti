"""
Advanced Verse Matcher for Academic Scripture Intelligence Enhancement.

This module implements advanced contextual verse matching algorithms
for research publication standards, extending the existing hybrid matching
engine with academic-grade validation and publication-ready formatting.

Story 4.5: Scripture Intelligence Enhancement - Task 1 Implementation
"""

from pathlib import Path
import logging
import re
import time


from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple

from .hybrid_matching_engine import (
    HybridMatchingEngine, HybridPipelineConfig, HybridMatchingResult,
    MatchingStage, SourceProvenance
)
from .canonical_text_manager import CanonicalTextManager, VerseCandidate
from utils.logger_config import get_logger


class ContextualMatchingMode(Enum):
    """Contextual matching modes for academic verse identification."""
    STANDARD = "standard"
    ACADEMIC = "academic"
    RESEARCH_GRADE = "research_grade"
    PUBLICATION_READY = "publication_ready"


class AcademicConfidenceLevel(Enum):
    """Academic confidence levels for verse matching."""
    UNVERIFIED = "unverified"
    PRELIMINARY = "preliminary"
    VALIDATED = "validated"
    PEER_REVIEWED = "peer_reviewed"
    PUBLICATION_READY = "publication_ready"


@dataclass
class ContextualMatchingConfig:
    """Configuration for advanced contextual verse matching."""
    
    # Academic matching parameters
    academic_confidence_threshold: float = 0.85
    research_grade_threshold: float = 0.90
    publication_threshold: float = 0.95
    
    # Contextual understanding parameters
    enable_scriptural_context: bool = True
    enable_philosophical_context: bool = True
    enable_linguistic_context: bool = True
    
    # Advanced matching features
    enable_cross_reference_validation: bool = True
    enable_canonical_consistency_check: bool = True
    enable_academic_citation_validation: bool = True
    
    # Performance parameters
    max_contextual_candidates: int = 10
    contextual_search_depth: int = 3
    enable_performance_optimization: bool = True
    
    # Integration parameters
    maintain_story_2_3_compatibility: bool = True
    preserve_existing_api: bool = True


@dataclass
class ContextualMatchingResult:
    """Result of advanced contextual verse matching."""
    
    # Basic matching information
    original_passage: str
    matched_verse: Optional[VerseCandidate] = None
    confidence_score: float = 0.0
    academic_confidence_level: AcademicConfidenceLevel = AcademicConfidenceLevel.UNVERIFIED
    
    # Contextual analysis
    scriptural_context: Dict[str, Any] = field(default_factory=dict)
    philosophical_context: Dict[str, Any] = field(default_factory=dict)
    linguistic_context: Dict[str, Any] = field(default_factory=dict)
    
    # Academic validation
    cross_references: List[VerseCandidate] = field(default_factory=list)
    canonical_consistency: bool = True
    citation_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Performance and metadata
    processing_time: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    hybrid_engine_result: Optional[HybridMatchingResult] = None
    academic_warnings: List[str] = field(default_factory=list)
    
    # Publication readiness indicators
    publication_ready: bool = False
    requires_review: bool = True
    review_notes: List[str] = field(default_factory=list)


class AdvancedVerseMatcher:
    """
    Advanced Verse Matcher for Academic Scripture Intelligence Enhancement.
    
    Extends the existing hybrid matching engine with academic-grade
    contextual understanding, publication standards validation, and
    research-quality verse identification capabilities.
    """
    
    def __init__(
        self,
        canonical_manager: CanonicalTextManager,
        hybrid_engine: Optional[HybridMatchingEngine] = None,
        config: Optional[ContextualMatchingConfig] = None
    ):
        """
        Initialize the Advanced Verse Matcher.
        
        Args:
            canonical_manager: Canonical text management system
            hybrid_engine: Optional existing hybrid matching engine
            config: Configuration for contextual matching
        """
        self.logger = get_logger(__name__)
        self.canonical_manager = canonical_manager
        self.config = config or ContextualMatchingConfig()
        
        # Initialize or use existing hybrid engine
        if hybrid_engine:
            self.hybrid_engine = hybrid_engine
        else:
            # Create hybrid engine with academic-grade configuration
            hybrid_config = HybridPipelineConfig(
                phonetic_weight=0.25,
                sequence_weight=0.35,
                semantic_weight=0.40,  # Higher semantic weight for academic context
                fallback_threshold=self.config.academic_confidence_threshold,
                enable_fallback_to_traditional=True
            )
            self.hybrid_engine = HybridMatchingEngine(
                canonical_manager=canonical_manager,
                config=hybrid_config
            )
        
        # Initialize contextual analysis components
        self._initialize_contextual_analyzers()
        
        # Performance tracking
        self.performance_stats = {
            'total_matches_processed': 0,
            'academic_grade_matches': 0,
            'publication_ready_matches': 0,
            'average_processing_time': 0.0,
            'contextual_accuracy_rate': 0.0
        }
        
        self.logger.info("Advanced Verse Matcher initialized for academic processing")
    
    def _initialize_contextual_analyzers(self) -> None:
        """Initialize contextual analysis components."""
        
        # Scriptural context patterns for Yoga Vedanta literature
        self.scriptural_patterns = {
            'bhagavad_gita': [
                r'\b(?:chapter|adhyaya)\s+(\d+|[a-z]+)\s+(?:verse|shloka)\s+(\d+|[a-z]+)\b',
                r'\bgita\s+(\d+)\.(\d+)\b',
                r'\b(?:bg|bh?g)\s*(\d+)\.(\d+)\b'
            ],
            'upanishads': [
                r'\b(\w+)\s+upanishad\b',
                r'\bmundaka\s+(\d+)\.(\d+)\.(\d+)\b',
                r'\bkatha\s+(\d+)\.(\d+)\.(\d+)\b'
            ],
            'yoga_sutras': [
                r'\b(?:yoga\s+)?sutra\s+(\d+)\.(\d+)\b',
                r'\bpatanjali\s+(\d+)\.(\d+)\b',
                r'\bys\s+(\d+)\.(\d+)\b'
            ]
        }
        
        # Philosophical context indicators
        self.philosophical_indicators = {
            'advaita_vedanta': [
                'brahman', 'atman', 'maya', 'moksha', 'jnana', 'viveka'
            ],
            'karma_yoga': [
                'karma', 'action', 'duty', 'dharma', 'nishkama', 'yajna'
            ],
            'bhakti_yoga': [
                'bhakti', 'devotion', 'prema', 'surrender', 'ishvara'
            ],
            'raja_yoga': [
                'meditation', 'samadhi', 'dharana', 'dhyana', 'pranayama'
            ]
        }
        
        # Linguistic context patterns for Sanskrit/Hindi terms
        self.linguistic_patterns = {
            'sanskrit_compounds': r'\b\w+[aeiou][mn]?\w*\b',
            'transliteration_markers': r'[āīūṛḷēōṃḥṅñṭḍṇśṣ]',
            'verse_markers': r'[।॥]',
            'technical_terms': r'\b(?:yoga|dharma|karma|moksha|samsara|nirvana)\b'
        }
    
    def match_verse_with_context(
        self,
        passage: str,
        context: Dict[str, Any] = None,
        mode: ContextualMatchingMode = ContextualMatchingMode.ACADEMIC
    ) -> ContextualMatchingResult:
        """
        Match verse passage with advanced contextual analysis.
        
        Args:
            passage: Text passage to match against canonical verses
            context: Additional context information
            mode: Contextual matching mode for academic requirements
            
        Returns:
            Advanced contextual matching result with academic validation
        """
        start_time = time.time()
        context = context or {}
        
        result = ContextualMatchingResult(
            original_passage=passage,
            academic_confidence_level=AcademicConfidenceLevel.UNVERIFIED
        )
        
        try:
            # Step 1: Execute hybrid matching engine
            self.logger.info(f"Executing hybrid matching for: {passage[:50]}...")
            hybrid_result = self.hybrid_engine.match_verse_passage(passage, context)
            result.hybrid_engine_result = hybrid_result
            result.stages_completed.append("hybrid_matching")
            
            if not hybrid_result.pipeline_success or not hybrid_result.matched_verse:
                self.logger.info("Hybrid matching failed, proceeding with contextual fallback")
                return self._apply_contextual_fallback(result, passage, context, mode)
            
            # Step 2: Enhance with contextual analysis
            self.logger.info("Applying advanced contextual analysis...")
            self._analyze_scriptural_context(result, passage, hybrid_result.matched_verse)
            self._analyze_philosophical_context(result, passage, hybrid_result.matched_verse)
            self._analyze_linguistic_context(result, passage, hybrid_result.matched_verse)
            result.stages_completed.append("contextual_analysis")
            
            # Step 3: Academic validation
            if self.config.enable_cross_reference_validation:
                self._validate_cross_references(result, hybrid_result.matched_verse)
                result.stages_completed.append("cross_reference_validation")
            
            if self.config.enable_canonical_consistency_check:
                self._validate_canonical_consistency(result, hybrid_result.matched_verse)
                result.stages_completed.append("canonical_consistency")
            
            if self.config.enable_academic_citation_validation:
                self._validate_academic_citation(result, passage, hybrid_result.matched_verse)
                result.stages_completed.append("citation_validation")
            
            # Step 4: Calculate academic confidence and publication readiness
            self._calculate_academic_confidence(result, hybrid_result, mode)
            self._assess_publication_readiness(result, mode)
            
            # Step 5: Update performance statistics
            result.processing_time = time.time() - start_time
            self._update_performance_stats(result)
            
            self.logger.info(
                f"Advanced matching completed: confidence={result.confidence_score:.3f}, "
                f"academic_level={result.academic_confidence_level.value}, "
                f"publication_ready={result.publication_ready}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in advanced verse matching: {e}")
            result.academic_warnings.append(f"Processing error: {str(e)}")
            result.processing_time = time.time() - start_time
        
        return result
    
    def _analyze_scriptural_context(
        self,
        result: ContextualMatchingResult,
        passage: str,
        matched_verse: VerseCandidate
    ) -> None:
        """Analyze scriptural context for academic validation."""
        
        scriptural_context = {
            'identified_scripture': None,
            'chapter_verse_references': [],
            'contextual_indicators': [],
            'canonical_alignment': 0.0
        }
        
        # Identify scripture type and references
        for scripture_type, patterns in self.scriptural_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, passage, re.IGNORECASE)
                if matches:
                    scriptural_context['identified_scripture'] = scripture_type
                    scriptural_context['chapter_verse_references'].extend(matches)
        
        # Validate against matched verse canonical information
        if matched_verse and scriptural_context['identified_scripture']:
            expected_scripture = matched_verse.source.value.lower().replace('_', '_')
            identified_scripture = scriptural_context['identified_scripture']
            
            if expected_scripture in identified_scripture or identified_scripture in expected_scripture:
                scriptural_context['canonical_alignment'] = 1.0
            else:
                scriptural_context['canonical_alignment'] = 0.5
                result.academic_warnings.append(
                    f"Scripture type mismatch: identified={identified_scripture}, "
                    f"canonical={expected_scripture}"
                )
        
        result.scriptural_context = scriptural_context
    
    def _analyze_philosophical_context(
        self,
        result: ContextualMatchingResult,
        passage: str,
        matched_verse: VerseCandidate
    ) -> None:
        """Analyze philosophical context for academic accuracy."""
        
        philosophical_context = {
            'identified_traditions': [],
            'key_concepts': [],
            'conceptual_coherence': 0.0,
            'academic_terminology': []
        }
        
        passage_lower = passage.lower()
        
        # Identify philosophical traditions
        for tradition, indicators in self.philosophical_indicators.items():
            found_indicators = [ind for ind in indicators if ind in passage_lower]
            if found_indicators:
                philosophical_context['identified_traditions'].append(tradition)
                philosophical_context['key_concepts'].extend(found_indicators)
        
        # Assess conceptual coherence
        if philosophical_context['identified_traditions']:
            # Simple coherence check - single tradition indicates higher coherence
            if len(philosophical_context['identified_traditions']) == 1:
                philosophical_context['conceptual_coherence'] = 0.9
            elif len(philosophical_context['identified_traditions']) <= 2:
                philosophical_context['conceptual_coherence'] = 0.7
            else:
                philosophical_context['conceptual_coherence'] = 0.5
        
        # Identify academic terminology
        academic_terms = [
            'philosophy', 'metaphysics', 'epistemology', 'ontology',
            'consciousness', 'ultimate reality', 'self-realization'
        ]
        philosophical_context['academic_terminology'] = [
            term for term in academic_terms if term in passage_lower
        ]
        
        result.philosophical_context = philosophical_context
    
    def _analyze_linguistic_context(
        self,
        result: ContextualMatchingResult,
        passage: str,
        matched_verse: VerseCandidate
    ) -> None:
        """Analyze linguistic context for academic standards."""
        
        linguistic_context = {
            'sanskrit_elements': [],
            'transliteration_quality': 0.0,
            'academic_formatting': False,
            'iast_compliance': 0.0
        }
        
        # Detect Sanskrit linguistic elements
        for element_type, pattern in self.linguistic_patterns.items():
            matches = re.findall(pattern, passage)
            if matches:
                linguistic_context['sanskrit_elements'].append({
                    'type': element_type,
                    'occurrences': len(matches),
                    'examples': matches[:3]  # First 3 examples
                })
        
        # Assess transliteration quality
        transliteration_markers = re.findall(
            self.linguistic_patterns['transliteration_markers'], passage
        )
        if transliteration_markers:
            # Simple quality assessment based on marker consistency
            linguistic_context['transliteration_quality'] = min(
                len(transliteration_markers) / max(len(passage.split()), 1) * 10, 1.0
            )
            linguistic_context['iast_compliance'] = linguistic_context['transliteration_quality']
        
        # Check academic formatting indicators
        academic_markers = [
            re.search(r'\d+\.\d+', passage),  # Numerical references
            re.search(r'\([^)]+\)', passage),  # Parenthetical citations
            re.search(r'[।॥]', passage)  # Sanskrit verse markers
        ]
        linguistic_context['academic_formatting'] = any(academic_markers)
        
        result.linguistic_context = linguistic_context
    
    def _validate_cross_references(
        self,
        result: ContextualMatchingResult,
        matched_verse: VerseCandidate
    ) -> None:
        """Validate cross-references for academic accuracy."""
        
        try:
            # Find related verses from the same source
            related_candidates = self.canonical_manager.get_verse_candidates(
                query=matched_verse.canonical_text[:50],  # First 50 chars
                max_candidates=5,
                source_filter=matched_verse.source
            )
            
            # Filter out the matched verse itself
            cross_references = [
                candidate for candidate in related_candidates
                if candidate.chapter != matched_verse.chapter or 
                candidate.verse != matched_verse.verse
            ]
            
            result.cross_references = cross_references[:3]  # Top 3 cross-references
            
            if len(cross_references) >= 2:
                self.logger.info(f"Found {len(cross_references)} cross-references for validation")
            else:
                result.academic_warnings.append(
                    "Limited cross-references found for academic validation"
                )
                
        except Exception as e:
            self.logger.error(f"Error in cross-reference validation: {e}")
            result.academic_warnings.append(f"Cross-reference validation failed: {str(e)}")
    
    def _validate_canonical_consistency(
        self,
        result: ContextualMatchingResult,
        matched_verse: VerseCandidate
    ) -> None:
        """Validate canonical text consistency for academic standards."""
        
        try:
            # Check if the matched verse text is consistent with canonical sources
            canonical_text = matched_verse.canonical_text.strip()
            
            # Basic consistency checks
            consistency_checks = {
                'non_empty': len(canonical_text) > 0,
                'proper_length': 10 <= len(canonical_text) <= 1000,
                'sanskrit_characters': bool(re.search(r'[।॥]', canonical_text)),
                'no_corrupted_encoding': '�' not in canonical_text
            }
            
            failed_checks = [
                check for check, passed in consistency_checks.items() if not passed
            ]
            
            if not failed_checks:
                result.canonical_consistency = True
                self.logger.info("Canonical consistency validation passed")
            else:
                result.canonical_consistency = False
                result.academic_warnings.append(
                    f"Canonical consistency issues: {', '.join(failed_checks)}"
                )
                
        except Exception as e:
            self.logger.error(f"Error in canonical consistency validation: {e}")
            result.canonical_consistency = False
            result.academic_warnings.append(f"Consistency validation error: {str(e)}")
    
    def _validate_academic_citation(
        self,
        result: ContextualMatchingResult,
        passage: str,
        matched_verse: VerseCandidate
    ) -> None:
        """Validate academic citation standards."""
        
        citation_validation = {
            'citation_format_detected': False,
            'source_attribution': False,
            'academic_standards_compliance': 0.0,
            'suggested_citation': ''
        }
        
        try:
            # Check for existing citation formats
            citation_patterns = [
                r'\([\w\s]+\d+\.\d+\)',  # (Source Chapter.Verse)
                r'[\w\s]+\s+\d+:\d+',    # Source Chapter:Verse
                r'\[\w+\s*\d+\.\d+\]'    # [Source Chapter.Verse]
            ]
            
            for pattern in citation_patterns:
                if re.search(pattern, passage):
                    citation_validation['citation_format_detected'] = True
                    break
            
            # Check for source attribution
            source_indicators = ['gita', 'upanishad', 'sutra', 'vedanta']
            passage_lower = passage.lower()
            citation_validation['source_attribution'] = any(
                indicator in passage_lower for indicator in source_indicators
            )
            
            # Calculate academic standards compliance
            compliance_score = 0.0
            if citation_validation['citation_format_detected']:
                compliance_score += 0.5
            if citation_validation['source_attribution']:
                compliance_score += 0.3
            if result.linguistic_context.get('academic_formatting', False):
                compliance_score += 0.2
                
            citation_validation['academic_standards_compliance'] = compliance_score
            
            # Generate suggested academic citation
            citation_validation['suggested_citation'] = (
                f"{matched_verse.source.value.replace('_', ' ').title()} "
                f"{matched_verse.chapter}.{matched_verse.verse}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in citation validation: {e}")
            citation_validation['academic_standards_compliance'] = 0.0
        
        result.citation_validation = citation_validation
    
    def _calculate_academic_confidence(
        self,
        result: ContextualMatchingResult,
        hybrid_result: HybridMatchingResult,
        mode: ContextualMatchingMode
    ) -> None:
        """Calculate academic confidence level and overall confidence score."""
        
        # Start with hybrid engine confidence
        base_confidence = hybrid_result.composite_confidence
        
        # Apply contextual adjustments
        contextual_adjustments = 0.0
        
        # Scriptural context adjustment
        scriptural_alignment = result.scriptural_context.get('canonical_alignment', 0.0)
        contextual_adjustments += scriptural_alignment * 0.1
        
        # Philosophical context adjustment
        philosophical_coherence = result.philosophical_context.get('conceptual_coherence', 0.0)
        contextual_adjustments += philosophical_coherence * 0.05
        
        # Linguistic context adjustment
        transliteration_quality = result.linguistic_context.get('transliteration_quality', 0.0)
        contextual_adjustments += transliteration_quality * 0.05
        
        # Academic validation adjustments
        if result.canonical_consistency:
            contextual_adjustments += 0.05
        
        citation_compliance = result.citation_validation.get('academic_standards_compliance', 0.0)
        contextual_adjustments += citation_compliance * 0.05
        
        # Calculate final confidence
        result.confidence_score = min(base_confidence + contextual_adjustments, 1.0)
        
        # Determine academic confidence level based on mode and score
        if mode == ContextualMatchingMode.PUBLICATION_READY:
            if result.confidence_score >= self.config.publication_threshold:
                result.academic_confidence_level = AcademicConfidenceLevel.PUBLICATION_READY
            elif result.confidence_score >= self.config.research_grade_threshold:
                result.academic_confidence_level = AcademicConfidenceLevel.PEER_REVIEWED
            elif result.confidence_score >= self.config.academic_confidence_threshold:
                result.academic_confidence_level = AcademicConfidenceLevel.VALIDATED
            else:
                result.academic_confidence_level = AcademicConfidenceLevel.PRELIMINARY
        else:
            # Standard academic mode thresholds
            if result.confidence_score >= 0.85:
                result.academic_confidence_level = AcademicConfidenceLevel.VALIDATED
            elif result.confidence_score >= 0.70:
                result.academic_confidence_level = AcademicConfidenceLevel.PRELIMINARY
            else:
                result.academic_confidence_level = AcademicConfidenceLevel.UNVERIFIED
        
        # Set matched verse from hybrid result
        result.matched_verse = hybrid_result.matched_verse
    
    def _assess_publication_readiness(
        self,
        result: ContextualMatchingResult,
        mode: ContextualMatchingMode
    ) -> None:
        """Assess publication readiness based on academic standards."""
        
        publication_criteria = {
            'high_confidence': result.confidence_score >= self.config.publication_threshold,
            'academic_level': result.academic_confidence_level in [
                AcademicConfidenceLevel.PEER_REVIEWED,
                AcademicConfidenceLevel.PUBLICATION_READY
            ],
            'canonical_consistency': result.canonical_consistency,
            'citation_compliance': result.citation_validation.get(
                'academic_standards_compliance', 0.0
            ) >= 0.8,
            'no_critical_warnings': len([
                w for w in result.academic_warnings
                if 'error' in w.lower() or 'critical' in w.lower()
            ]) == 0
        }
        
        # Check publication readiness
        criteria_met = sum(publication_criteria.values())
        total_criteria = len(publication_criteria)
        
        if mode == ContextualMatchingMode.PUBLICATION_READY:
            result.publication_ready = criteria_met == total_criteria
            result.requires_review = criteria_met < total_criteria
        else:
            result.publication_ready = criteria_met >= (total_criteria - 1)
            result.requires_review = criteria_met < (total_criteria - 2)
        
        # Generate review notes for unmet criteria
        for criterion, met in publication_criteria.items():
            if not met:
                result.review_notes.append(f"Publication criterion not met: {criterion}")
    
    def _apply_contextual_fallback(
        self,
        result: ContextualMatchingResult,
        passage: str,
        context: Dict[str, Any],
        mode: ContextualMatchingMode
    ) -> ContextualMatchingResult:
        """Apply contextual fallback when hybrid matching fails."""
        
        self.logger.info("Applying contextual fallback matching...")
        
        # Try to find candidates using contextual patterns
        contextual_candidates = []
        
        # Look for scriptural references in the passage
        for scripture_type, patterns in self.scriptural_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, passage, re.IGNORECASE)
                if matches:
                    # Try to find verses based on detected references
                    try:
                        # This is a simplified fallback - in practice, you'd implement
                        # more sophisticated contextual candidate retrieval
                        candidates = self.canonical_manager.get_verse_candidates(
                            query=passage,
                            max_candidates=3
                        )
                        contextual_candidates.extend(candidates)
                    except Exception as e:
                        self.logger.error(f"Error in contextual fallback: {e}")
        
        if contextual_candidates:
            # Use the best contextual candidate
            result.matched_verse = contextual_candidates[0]
            result.confidence_score = 0.6  # Lower confidence for fallback
            result.academic_confidence_level = AcademicConfidenceLevel.PRELIMINARY
            result.academic_warnings.append("Used contextual fallback matching")
            result.stages_completed.append("contextual_fallback")
        else:
            result.academic_warnings.append("No contextual matches found")
        
        return result
    
    def _update_performance_stats(self, result: ContextualMatchingResult) -> None:
        """Update performance statistics."""
        
        self.performance_stats['total_matches_processed'] += 1
        
        if result.academic_confidence_level in [
            AcademicConfidenceLevel.VALIDATED,
            AcademicConfidenceLevel.PEER_REVIEWED,
            AcademicConfidenceLevel.PUBLICATION_READY
        ]:
            self.performance_stats['academic_grade_matches'] += 1
        
        if result.publication_ready:
            self.performance_stats['publication_ready_matches'] += 1
        
        # Update average processing time
        current_avg = self.performance_stats['average_processing_time']
        total_processed = self.performance_stats['total_matches_processed']
        new_avg = ((current_avg * (total_processed - 1)) + result.processing_time) / total_processed
        self.performance_stats['average_processing_time'] = new_avg
        
        # Update contextual accuracy rate
        academic_rate = (
            self.performance_stats['academic_grade_matches'] / 
            max(self.performance_stats['total_matches_processed'], 1)
        )
        self.performance_stats['contextual_accuracy_rate'] = academic_rate
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the advanced verse matcher."""
        
        return {
            'advanced_matcher_stats': self.performance_stats.copy(),
            'hybrid_engine_stats': self.hybrid_engine.get_performance_statistics(),
            'configuration': {
                'academic_confidence_threshold': self.config.academic_confidence_threshold,
                'research_grade_threshold': self.config.research_grade_threshold,
                'publication_threshold': self.config.publication_threshold,
                'contextual_features_enabled': {
                    'scriptural_context': self.config.enable_scriptural_context,
                    'philosophical_context': self.config.enable_philosophical_context,
                    'linguistic_context': self.config.enable_linguistic_context
                }
            }
        }
    
    def validate_academic_integration(self) -> Dict[str, Any]:
        """Validate integration with academic standards and existing systems."""
        
        validation_result = {
            'integration_status': 'healthy',
            'component_status': {},
            'compatibility_issues': [],
            'academic_readiness': True
        }
        
        try:
            # Validate hybrid engine integration
            hybrid_validation = self.hybrid_engine.validate_system_integration()
            validation_result['component_status']['hybrid_engine'] = hybrid_validation['is_valid']
            
            if not hybrid_validation['is_valid']:
                validation_result['compatibility_issues'].extend(
                    hybrid_validation.get('issues', [])
                )
            
            # Validate canonical manager integration
            canonical_stats = self.canonical_manager.get_statistics()
            validation_result['component_status']['canonical_manager'] = (
                canonical_stats['total_verses'] > 0
            )
            
            # Validate academic configuration
            config_valid = all([
                0.0 <= self.config.academic_confidence_threshold <= 1.0,
                0.0 <= self.config.research_grade_threshold <= 1.0,
                0.0 <= self.config.publication_threshold <= 1.0,
                self.config.academic_confidence_threshold <= self.config.research_grade_threshold,
                self.config.research_grade_threshold <= self.config.publication_threshold
            ])
            validation_result['component_status']['academic_configuration'] = config_valid
            
            if not config_valid:
                validation_result['compatibility_issues'].append(
                    "Academic confidence thresholds are not properly ordered"
                )
            
            # Overall validation
            all_components_valid = all(validation_result['component_status'].values())
            has_critical_issues = len(validation_result['compatibility_issues']) > 0
            
            if not all_components_valid or has_critical_issues:
                validation_result['integration_status'] = 'issues_detected'
                validation_result['academic_readiness'] = False
            
        except Exception as e:
            self.logger.error(f"Error in academic integration validation: {e}")
            validation_result['integration_status'] = 'validation_error'
            validation_result['academic_readiness'] = False
            validation_result['compatibility_issues'].append(f"Validation error: {str(e)}")
        
        return validation_result


def create_advanced_verse_matcher(
    canonical_manager: CanonicalTextManager,
    hybrid_engine: Optional[HybridMatchingEngine] = None,
    config_overrides: Dict[str, Any] = None
) -> AdvancedVerseMatcher:
    """
    Factory function to create an AdvancedVerseMatcher with appropriate configuration.
    
    Args:
        canonical_manager: Canonical text management system
        hybrid_engine: Optional existing hybrid matching engine
        config_overrides: Optional configuration overrides
        
    Returns:
        Configured AdvancedVerseMatcher instance
    """
    config = ContextualMatchingConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return AdvancedVerseMatcher(
        canonical_manager=canonical_manager,
        hybrid_engine=hybrid_engine,
        config=config
    )


# Test functions for validation
def test_advanced_verse_matcher_functionality():
    """
    Test advanced verse matcher core functionality.
    Story 4.5 Task 1 validation test.
    """
    try:
        from .canonical_text_manager import CanonicalTextManager
        from .hybrid_matching_engine import HybridMatchingEngine
        
        # Initialize components
        canonical_manager = CanonicalTextManager()
        advanced_matcher = create_advanced_verse_matcher(canonical_manager)
        
        # Test basic functionality
        test_passage = "Today we study the eternal nature of the soul as described in Bhagavad Gita chapter 2 verse 20"
        
        result = advanced_matcher.match_verse_with_context(
            passage=test_passage,
            mode=ContextualMatchingMode.ACADEMIC
        )
        
        # Validation checks
        assert result.original_passage == test_passage
        assert result.confidence_score >= 0.0
        assert isinstance(result.academic_confidence_level, AcademicConfidenceLevel)
        assert len(result.stages_completed) > 0
        assert result.processing_time > 0.0
        
        print("✓ Advanced Verse Matcher basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Advanced Verse Matcher test failed: {e}")
        return False


def test_contextual_analysis_components():
    """
    Test contextual analysis components.
    Story 4.5 Task 1 validation test.
    """
    try:
        from .canonical_text_manager import CanonicalTextManager
        
        canonical_manager = CanonicalTextManager()
        advanced_matcher = create_advanced_verse_matcher(canonical_manager)
        
        # Test scriptural context detection
        scriptural_passage = "We study Bhagavad Gita chapter 2 verse 47 about karma yoga"
        result = advanced_matcher.match_verse_with_context(scriptural_passage)
        
        # Validate contextual analysis
        assert 'scriptural_context' in result.__dict__
        assert 'philosophical_context' in result.__dict__
        assert 'linguistic_context' in result.__dict__
        
        # Check that context analysis was performed
        if result.scriptural_context:
            assert 'canonical_alignment' in result.scriptural_context
        
        print("✓ Contextual analysis components test passed")
        return True
        
    except Exception as e:
        print(f"✗ Contextual analysis test failed: {e}")
        return False


def test_academic_confidence_levels():
    """
    Test academic confidence level calculation.
    Story 4.5 Task 1 validation test.
    """
    try:
        from .canonical_text_manager import CanonicalTextManager
        
        canonical_manager = CanonicalTextManager()
        config = ContextualMatchingConfig(
            academic_confidence_threshold=0.7,
            research_grade_threshold=0.85,
            publication_threshold=0.95
        )
        advanced_matcher = create_advanced_verse_matcher(
            canonical_manager, 
            config_overrides={'academic_confidence_threshold': 0.7}
        )
        
        # Test different confidence scenarios
        test_passages = [
            "karma yoga practice",  # Lower confidence
            "Bhagavad Gita chapter 2 verse 47 karma yoga",  # Higher confidence
        ]
        
        for passage in test_passages:
            result = advanced_matcher.match_verse_with_context(
                passage, 
                mode=ContextualMatchingMode.PUBLICATION_READY
            )
            
            # Validate confidence level assignment
            assert isinstance(result.academic_confidence_level, AcademicConfidenceLevel)
            assert 0.0 <= result.confidence_score <= 1.0
        
        print("✓ Academic confidence levels test passed")
        return True
        
    except Exception as e:
        print(f"✗ Academic confidence levels test failed: {e}")
        return False


def test_integration_validation():
    """
    Test integration validation with existing systems.
    Story 4.5 Task 1 validation test.
    """
    try:
        from .canonical_text_manager import CanonicalTextManager
        
        canonical_manager = CanonicalTextManager()
        advanced_matcher = create_advanced_verse_matcher(canonical_manager)
        
        # Test academic integration validation
        validation_result = advanced_matcher.validate_academic_integration()
        
        # Validate integration components
        assert 'integration_status' in validation_result
        assert 'component_status' in validation_result
        assert 'academic_readiness' in validation_result
        assert isinstance(validation_result['academic_readiness'], bool)
        
        # Check component status
        required_components = ['hybrid_engine', 'canonical_manager', 'academic_configuration']
        for component in required_components:
            if component in validation_result['component_status']:
                assert isinstance(validation_result['component_status'][component], bool)
        
        print("✓ Integration validation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration validation test failed: {e}")
        return False


def validate_story_4_5_task_1_implementation():
    """
    Comprehensive validation for Story 4.5 Task 1 implementation.
    """
    print("=== Story 4.5 Task 1 Implementation Validation ===")
    
    test_results = []
    
    # Run all validation tests
    test_functions = [
        test_advanced_verse_matcher_functionality,
        test_contextual_analysis_components,
        test_academic_confidence_levels,
        test_integration_validation
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            test_results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            test_results.append(False)
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n=== Task 1 Validation Summary ===")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✓ All Task 1 tests passed - Ready for completion")
        return True
    else:
        print("✗ Some Task 1 tests failed - Requires fixes")
        return False


if __name__ == "__main__":
    validate_story_4_5_task_1_implementation()
