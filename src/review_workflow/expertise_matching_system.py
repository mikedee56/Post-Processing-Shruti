"""
Epic 4.1 MCP Context-Aware Expertise Matching System.

Implements Story 3.3 Task 2: Epic 4.1 MCP Context-Aware Expertise Matching
- Intelligent reviewer-content matching using MCP framework
- Context-aware processing for sophisticated complexity rating
- Circuit breaker patterns for reliable assignment algorithms
- Epic 4.5 academic standards integration for expertise profiling
"""

import logging
import math
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union

# Epic 4 infrastructure imports
from utils.mcp_client_manager import MCPClientManager
from utils.mcp_transformer_client import create_transformer_client
from sanskrit_hindi_identifier.enhanced_lexicon_manager import EnhancedLexiconManager
from utils.performance_monitor import PerformanceMonitor
from monitoring.system_monitor import SystemMonitor
from monitoring.telemetry_collector import TelemetryCollector
from utils.srt_parser import SRTSegment

# Review workflow imports
from .review_workflow_engine import ReviewerProfile, ReviewerRole, ReviewerSkill


class ComplexityDimension(Enum):
    """Dimensions of content complexity for matching."""
    LINGUISTIC_COMPLEXITY = "linguistic_complexity"
    SANSKRIT_CONTENT_DENSITY = "sanskrit_content_density"
    ACADEMIC_RIGOR_REQUIRED = "academic_rigor_required"
    THEOLOGICAL_DEPTH = "theological_depth"
    TECHNICAL_TERMINOLOGY = "technical_terminology"
    CITATION_COMPLEXITY = "citation_complexity"


class MatchingConfidence(Enum):
    """Confidence levels for reviewer-content matching."""
    EXCELLENT = "excellent"  # 0.9+
    GOOD = "good"           # 0.7-0.89
    MODERATE = "moderate"   # 0.5-0.69
    POOR = "poor"          # 0.3-0.49
    UNFIT = "unfit"        # <0.3


@dataclass
class ComplexityRating:
    """Comprehensive content complexity assessment using Epic 4.1 MCP."""
    content_id: str
    overall_complexity: float  # 0.0 - 1.0
    
    # Dimension-specific ratings
    dimension_scores: Dict[ComplexityDimension, float] = field(default_factory=dict)
    
    # Content analysis details
    sanskrit_term_count: int = 0
    theological_concept_count: int = 0
    academic_reference_count: int = 0
    unique_vocabulary_ratio: float = 0.0
    
    # MCP-enhanced analysis (Epic 4.1)
    mcp_context_analysis: Dict[str, Any] = field(default_factory=dict)
    semantic_complexity_score: float = 0.0
    contextual_coherence_score: float = 0.0
    
    # Confidence and metadata
    analysis_confidence: float = 1.0
    processing_time_ms: float = 0.0
    circuit_breaker_used: bool = False
    
    # Academic standards (Epic 4.5)
    requires_specialist: bool = False
    recommended_expertise_level: str = "intermediate"
    academic_validation_required: bool = False


@dataclass
class ReviewerMatch:
    """Reviewer-content matching result with Epic 4.1 context awareness."""
    reviewer_id: str
    content_complexity: ComplexityRating
    
    # Matching analysis
    overall_match_score: float  # 0.0 - 1.0
    confidence_level: MatchingConfidence
    
    # Detailed matching scores
    skill_alignment_score: float = 0.0
    experience_match_score: float = 0.0
    workload_feasibility_score: float = 0.0
    specialization_bonus: float = 0.0
    
    # MCP-enhanced matching (Epic 4.1)
    context_awareness_score: float = 0.0
    semantic_compatibility_score: float = 0.0
    cultural_context_match: float = 0.0
    
    # Risk assessment
    potential_issues: List[str] = field(default_factory=list)
    estimated_review_time_hours: float = 0.0
    quality_risk_level: str = "low"
    
    # Academic considerations (Epic 4.5)
    academic_qualification_match: float = 0.0
    publication_readiness_capability: bool = False
    consultant_escalation_likelihood: float = 0.0


class ExpertiseMatchingSystem:
    """
    Epic 4.1 MCP Context-Aware Expertise Matching System.
    
    Implements Story 3.3 Task 2:
    - Intelligent reviewer-content matching using MCP framework
    - Context-aware processing for sophisticated complexity rating
    - Circuit breaker patterns for reliable assignment algorithms  
    - Epic 4.5 academic standards for expertise profiling and skill assessment
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize expertise matching system with Epic 4.1 MCP integration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Epic 4.1 MCP Infrastructure
        self.mcp_client_manager = MCPClientManager(
            self.config.get('mcp_client', {})
        )
        self.mcp_transformer = create_transformer_client()
        self.enhanced_lexicon = EnhancedLexiconManager()
        
        # Epic 4.3 Production Infrastructure
        self.system_monitor = SystemMonitor(self.config.get('monitoring', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('performance', {}))
        self.telemetry_collector = TelemetryCollector(self.config.get('telemetry', {}))
        
        # Matching system configuration
        self.matching_config = self.config.get('matching', {
            'mcp_analysis_timeout_ms': 2000,
            'complexity_analysis_cache_ttl_hours': 24,
            'min_match_score_threshold': 0.3,
            'excellent_match_threshold': 0.9,
            'good_match_threshold': 0.7,
            'workload_weight': 0.2,
            'skill_weight': 0.4,
            'experience_weight': 0.3,
            'specialization_weight': 0.1
        })
        
        # Data structures
        self.complexity_cache: Dict[str, ComplexityRating] = {}
        self.reviewer_performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self.matching_statistics = defaultdict(int)
        
        # Threading and reliability (Epic 4.3)
        self.lock = threading.RLock()
        
        # Epic 4.1 Circuit breaker pattern
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_open = False
        self.circuit_breaker_reset_time = None
        
        # Performance tracking
        self.analysis_times = deque(maxlen=100)
        self.match_success_rates = deque(maxlen=100)
        
        self.logger.info("ExpertiseMatchingSystem initialized with Epic 4.1 MCP integration")
    
    def analyze_content_complexity(self, 
                                 content_id: str,
                                 segments: List[SRTSegment],
                                 use_cache: bool = True) -> ComplexityRating:
        """
        Analyze content complexity using Epic 4.1 MCP context-aware processing.
        
        Args:
            content_id: Unique identifier for content
            segments: SRT segments to analyze
            use_cache: Whether to use cached analysis
            
        Returns:
            ComplexityRating: Comprehensive complexity assessment
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("analyze_content_complexity"):
            # Check cache first
            if use_cache and content_id in self.complexity_cache:
                cached_rating = self.complexity_cache[content_id]
                if (datetime.now() - datetime.fromtimestamp(cached_rating.processing_time_ms / 1000)).total_seconds() < \
                   self.matching_config['complexity_analysis_cache_ttl_hours'] * 3600:
                    return cached_rating
            
            # Circuit breaker check (Epic 4.1)
            if self._check_circuit_breaker():
                return self._create_fallback_complexity_rating(content_id, segments)
            
            try:
                # Combine all segment text for analysis
                combined_text = " ".join([segment.text for segment in segments])
                
                # Basic linguistic analysis
                basic_analysis = self._perform_basic_complexity_analysis(combined_text)
                
                # Epic 4.1 MCP-enhanced analysis
                mcp_analysis = self._perform_mcp_complexity_analysis(combined_text)
                
                # Calculate dimension-specific scores
                dimension_scores = self._calculate_dimension_scores(combined_text, basic_analysis, mcp_analysis)
                
                # Calculate overall complexity
                overall_complexity = self._calculate_overall_complexity(dimension_scores)
                
                # Create complexity rating
                rating = ComplexityRating(
                    content_id=content_id,
                    overall_complexity=overall_complexity,
                    dimension_scores=dimension_scores,
                    sanskrit_term_count=basic_analysis['sanskrit_terms'],
                    theological_concept_count=basic_analysis['theological_concepts'],
                    academic_reference_count=basic_analysis['academic_references'],
                    unique_vocabulary_ratio=basic_analysis['unique_vocabulary_ratio'],
                    mcp_context_analysis=mcp_analysis,
                    semantic_complexity_score=mcp_analysis.get('semantic_complexity', 0.5),
                    contextual_coherence_score=mcp_analysis.get('contextual_coherence', 0.5),
                    analysis_confidence=mcp_analysis.get('confidence', 1.0),
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
                # Determine academic requirements
                self._assess_academic_requirements(rating)
                
                # Cache result
                self.complexity_cache[content_id] = rating
                
                # Record successful analysis
                self.circuit_breaker_failures = 0
                self.analysis_times.append(rating.processing_time_ms)
                
                # Telemetry
                self.telemetry_collector.record_event("complexity_analysis_completed", {
                    'content_id': content_id,
                    'overall_complexity': overall_complexity,
                    'processing_time_ms': rating.processing_time_ms,
                    'mcp_analysis_used': True
                })
                
                self.matching_statistics['complexity_analyses'] += 1
                
                return rating
                
            except Exception as e:
                # Circuit breaker increment
                self.circuit_breaker_failures += 1
                if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
                    self.circuit_breaker_open = True
                    self.circuit_breaker_reset_time = time.time() + 300  # 5 minute reset
                
                self.logger.error(f"Complexity analysis failed: {e}")
                return self._create_fallback_complexity_rating(content_id, segments)
    
    def find_best_reviewer_match(self,
                               complexity_rating: ComplexityRating,
                               available_reviewers: List[ReviewerProfile],
                               review_priority: str = "standard") -> Optional[ReviewerMatch]:
        """
        Find best reviewer match using Epic 4.1 context-aware matching.
        
        Args:
            complexity_rating: Content complexity assessment
            available_reviewers: List of available reviewers
            review_priority: Priority level for matching
            
        Returns:
            ReviewerMatch: Best matching reviewer or None
        """
        start_time = time.time()
        
        with self.performance_monitor.monitor_processing_operation("find_best_reviewer_match"):
            if not available_reviewers:
                return None
            
            best_match = None
            best_score = 0.0
            
            for reviewer in available_reviewers:
                try:
                    # Calculate comprehensive match score
                    match = self._calculate_reviewer_match(complexity_rating, reviewer, review_priority)
                    
                    if match.overall_match_score > best_score and \
                       match.overall_match_score >= self.matching_config['min_match_score_threshold']:
                        best_match = match
                        best_score = match.overall_match_score
                        
                except Exception as e:
                    self.logger.error(f"Error calculating match for reviewer {reviewer.reviewer_id}: {e}")
                    continue
            
            # Record matching statistics
            processing_time = time.time() - start_time
            success = best_match is not None
            
            self.match_success_rates.append(1.0 if success else 0.0)
            self.matching_statistics['matching_attempts'] += 1
            if success:
                self.matching_statistics['successful_matches'] += 1
            
            # Telemetry
            self.telemetry_collector.record_event("reviewer_matching_completed", {
                'complexity_overall': complexity_rating.overall_complexity,
                'reviewers_evaluated': len(available_reviewers),
                'match_found': success,
                'best_score': best_score if best_match else 0.0,
                'processing_time_ms': processing_time * 1000
            })
            
            if best_match:
                self.logger.info(f"Best reviewer match found: {best_match.reviewer_id} (score: {best_score:.3f})")
            else:
                self.logger.warning("No suitable reviewer match found")
            
            return best_match
    
    def _perform_basic_complexity_analysis(self, text: str) -> Dict[str, Any]:
        """Perform basic linguistic complexity analysis."""
        words = text.lower().split()
        unique_words = set(words)
        
        # Count Sanskrit/Hindi terms
        sanskrit_terms = sum(1 for word in words if any(term in word for term in [
            'yoga', 'dharma', 'karma', 'vedanta', 'upanishad', 'gita', 'krishna', 'arjuna', 'bhagavad'
        ]))
        
        # Count theological concepts
        theological_concepts = sum(1 for word in words if any(concept in word for concept in [
            'moksha', 'nirvana', 'samadhi', 'atman', 'brahman', 'samsara', 'ahimsa'
        ]))
        
        # Count potential academic references
        academic_references = text.count('(') + text.count('[')  # Basic citation pattern detection
        
        return {
            'total_words': len(words),
            'unique_words': len(unique_words),
            'unique_vocabulary_ratio': len(unique_words) / max(len(words), 1),
            'sanskrit_terms': sanskrit_terms,
            'theological_concepts': theological_concepts,
            'academic_references': academic_references,
            'average_word_length': statistics.mean([len(word) for word in words]) if words else 0
        }
    
    def _perform_mcp_complexity_analysis(self, text: str) -> Dict[str, Any]:
        """Perform Epic 4.1 MCP-enhanced complexity analysis."""
        try:
            # Use MCP transformer for semantic analysis
            mcp_result = self.mcp_transformer.analyze_text_complexity(
                text=text,
                timeout_ms=self.matching_config['mcp_analysis_timeout_ms']
            )
            
            if mcp_result and mcp_result.get('success', False):
                return {
                    'semantic_complexity': mcp_result.get('semantic_complexity', 0.5),
                    'contextual_coherence': mcp_result.get('contextual_coherence', 0.5),
                    'cultural_context_depth': mcp_result.get('cultural_context_depth', 0.5),
                    'terminology_sophistication': mcp_result.get('terminology_sophistication', 0.5),
                    'academic_register_level': mcp_result.get('academic_register_level', 0.5),
                    'confidence': mcp_result.get('confidence', 1.0),
                    'analysis_method': 'mcp_transformer'
                }
            else:
                # Fallback to rule-based analysis
                return self._fallback_complexity_analysis(text)
                
        except Exception as e:
            self.logger.warning(f"MCP complexity analysis failed, using fallback: {e}")
            return self._fallback_complexity_analysis(text)
    
    def _fallback_complexity_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback complexity analysis when MCP is unavailable."""
        # Simple rule-based analysis
        text_lower = text.lower()
        
        # Estimate semantic complexity based on vocabulary
        complex_words = sum(1 for word in text.split() if len(word) > 8)
        total_words = len(text.split())
        semantic_complexity = min(complex_words / max(total_words, 1) * 2, 1.0)
        
        # Estimate contextual coherence based on text structure
        sentences = text.count('.') + text.count('!') + text.count('?')
        contextual_coherence = min(sentences / max(total_words / 20, 1), 1.0)
        
        return {
            'semantic_complexity': semantic_complexity,
            'contextual_coherence': contextual_coherence,
            'cultural_context_depth': 0.5,  # Default neutral
            'terminology_sophistication': 0.5,
            'academic_register_level': 0.5,
            'confidence': 0.7,  # Lower confidence for fallback
            'analysis_method': 'rule_based_fallback'
        }
    
    def _calculate_dimension_scores(self, text: str, basic_analysis: Dict, mcp_analysis: Dict) -> Dict[ComplexityDimension, float]:
        """Calculate complexity scores for each dimension."""
        scores = {}
        
        # Linguistic complexity
        scores[ComplexityDimension.LINGUISTIC_COMPLEXITY] = min(
            (basic_analysis['average_word_length'] / 10) + 
            (basic_analysis['unique_vocabulary_ratio'] * 0.5) +
            (mcp_analysis['semantic_complexity'] * 0.5), 1.0
        )
        
        # Sanskrit content density
        scores[ComplexityDimension.SANSKRIT_CONTENT_DENSITY] = min(
            basic_analysis['sanskrit_terms'] / max(basic_analysis['total_words'] / 20, 1), 1.0
        )
        
        # Academic rigor required
        scores[ComplexityDimension.ACADEMIC_RIGOR_REQUIRED] = min(
            (basic_analysis['academic_references'] / max(basic_analysis['total_words'] / 50, 1)) +
            (mcp_analysis['academic_register_level'] * 0.7), 1.0
        )
        
        # Theological depth
        scores[ComplexityDimension.THEOLOGICAL_DEPTH] = min(
            (basic_analysis['theological_concepts'] / max(basic_analysis['total_words'] / 30, 1)) +
            (mcp_analysis['cultural_context_depth'] * 0.6), 1.0
        )
        
        # Technical terminology
        scores[ComplexityDimension.TECHNICAL_TERMINOLOGY] = mcp_analysis['terminology_sophistication']
        
        # Citation complexity
        scores[ComplexityDimension.CITATION_COMPLEXITY] = min(
            basic_analysis['academic_references'] / max(basic_analysis['total_words'] / 100, 1), 1.0
        )
        
        return scores
    
    def _calculate_overall_complexity(self, dimension_scores: Dict[ComplexityDimension, float]) -> float:
        """Calculate overall complexity from dimension scores."""
        if not dimension_scores:
            return 0.5
        
        # Weighted average of dimension scores
        weights = {
            ComplexityDimension.LINGUISTIC_COMPLEXITY: 0.15,
            ComplexityDimension.SANSKRIT_CONTENT_DENSITY: 0.25,
            ComplexityDimension.ACADEMIC_RIGOR_REQUIRED: 0.20,
            ComplexityDimension.THEOLOGICAL_DEPTH: 0.20,
            ComplexityDimension.TECHNICAL_TERMINOLOGY: 0.15,
            ComplexityDimension.CITATION_COMPLEXITY: 0.05
        }
        
        weighted_sum = sum(
            dimension_scores.get(dimension, 0.5) * weight
            for dimension, weight in weights.items()
        )
        
        return min(weighted_sum, 1.0)
    
    def _assess_academic_requirements(self, rating: ComplexityRating) -> None:
        """Assess academic requirements based on complexity rating."""
        # Requires specialist if high complexity in key dimensions
        rating.requires_specialist = (
            rating.dimension_scores.get(ComplexityDimension.ACADEMIC_RIGOR_REQUIRED, 0) > 0.7 or
            rating.dimension_scores.get(ComplexityDimension.THEOLOGICAL_DEPTH, 0) > 0.8 or
            rating.overall_complexity > 0.85
        )
        
        # Determine expertise level needed
        if rating.overall_complexity > 0.8:
            rating.recommended_expertise_level = "expert"
        elif rating.overall_complexity > 0.6:
            rating.recommended_expertise_level = "advanced"
        elif rating.overall_complexity > 0.4:
            rating.recommended_expertise_level = "intermediate"
        else:
            rating.recommended_expertise_level = "novice"
        
        # Academic validation required for high academic rigor
        rating.academic_validation_required = (
            rating.dimension_scores.get(ComplexityDimension.ACADEMIC_RIGOR_REQUIRED, 0) > 0.6
        )
    
    def _calculate_reviewer_match(self, 
                                complexity_rating: ComplexityRating,
                                reviewer: ReviewerProfile,
                                priority: str) -> ReviewerMatch:
        """Calculate comprehensive reviewer-content match score."""
        
        # 1. Skill alignment score
        skill_score = self._calculate_skill_alignment(complexity_rating, reviewer)
        
        # 2. Experience match score
        experience_score = self._calculate_experience_match(complexity_rating, reviewer)
        
        # 3. Workload feasibility score
        workload_score = self._calculate_workload_feasibility(reviewer, priority)
        
        # 4. Specialization bonus
        specialization_bonus = self._calculate_specialization_bonus(complexity_rating, reviewer)
        
        # 5. Epic 4.1 MCP-enhanced scores
        context_awareness_score = self._calculate_context_awareness(complexity_rating, reviewer)
        semantic_compatibility_score = self._calculate_semantic_compatibility(complexity_rating, reviewer)
        cultural_context_match = self._calculate_cultural_context_match(complexity_rating, reviewer)
        
        # 6. Academic qualification match (Epic 4.5)
        academic_qualification_match = self._calculate_academic_qualification_match(complexity_rating, reviewer)
        
        # Calculate overall match score with weights
        overall_score = (
            skill_score * self.matching_config['skill_weight'] +
            experience_score * self.matching_config['experience_weight'] +
            workload_score * self.matching_config['workload_weight'] +
            specialization_bonus * self.matching_config['specialization_weight'] +
            context_awareness_score * 0.1 +
            semantic_compatibility_score * 0.1 +
            cultural_context_match * 0.1 +
            academic_qualification_match * 0.1
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_score)
        
        # Calculate estimated review time
        estimated_time = self._estimate_review_time(complexity_rating, reviewer)
        
        # Assess potential issues and risks
        potential_issues = self._assess_potential_issues(complexity_rating, reviewer)
        quality_risk = self._assess_quality_risk(overall_score, complexity_rating, reviewer)
        
        # Calculate consultant escalation likelihood
        escalation_likelihood = self._calculate_escalation_likelihood(complexity_rating, reviewer)
        
        return ReviewerMatch(
            reviewer_id=reviewer.reviewer_id,
            content_complexity=complexity_rating,
            overall_match_score=overall_score,
            confidence_level=confidence_level,
            skill_alignment_score=skill_score,
            experience_match_score=experience_score,
            workload_feasibility_score=workload_score,
            specialization_bonus=specialization_bonus,
            context_awareness_score=context_awareness_score,
            semantic_compatibility_score=semantic_compatibility_score,
            cultural_context_match=cultural_context_match,
            academic_qualification_match=academic_qualification_match,
            potential_issues=potential_issues,
            estimated_review_time_hours=estimated_time,
            quality_risk_level=quality_risk,
            publication_readiness_capability=reviewer.role in [ReviewerRole.ACADEMIC_CONSULTANT, ReviewerRole.SENIOR_REVIEWER],
            consultant_escalation_likelihood=escalation_likelihood
        )
    
    def _calculate_skill_alignment(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> float:
        """Calculate how well reviewer skills align with content complexity."""
        if not reviewer.skills:
            return 0.3  # Default for reviewers without detailed skill profiles
        
        alignment_scores = []
        
        for skill in reviewer.skills:
            # Match skills to complexity dimensions
            if 'sanskrit' in skill.skill_area.lower():
                dimension_score = complexity_rating.dimension_scores.get(ComplexityDimension.SANSKRIT_CONTENT_DENSITY, 0.5)
                alignment = min(skill.proficiency_level / dimension_score, 1.0) if dimension_score > 0 else 1.0
                alignment_scores.append(alignment * 0.3)  # High weight for Sanskrit skills
            
            elif 'academic' in skill.skill_area.lower() or 'research' in skill.skill_area.lower():
                dimension_score = complexity_rating.dimension_scores.get(ComplexityDimension.ACADEMIC_RIGOR_REQUIRED, 0.5)
                alignment = min(skill.proficiency_level / dimension_score, 1.0) if dimension_score > 0 else 1.0
                alignment_scores.append(alignment * 0.25)
            
            elif 'theological' in skill.skill_area.lower() or 'vedanta' in skill.skill_area.lower():
                dimension_score = complexity_rating.dimension_scores.get(ComplexityDimension.THEOLOGICAL_DEPTH, 0.5)
                alignment = min(skill.proficiency_level / dimension_score, 1.0) if dimension_score > 0 else 1.0
                alignment_scores.append(alignment * 0.25)
            
            else:
                # General linguistic or editing skills
                dimension_score = complexity_rating.dimension_scores.get(ComplexityDimension.LINGUISTIC_COMPLEXITY, 0.5)
                alignment = min(skill.proficiency_level / dimension_score, 1.0) if dimension_score > 0 else 1.0
                alignment_scores.append(alignment * 0.2)
        
        return min(sum(alignment_scores), 1.0) if alignment_scores else 0.3
    
    def _calculate_experience_match(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> float:
        """Calculate experience level match with content complexity."""
        # Base experience score from review history
        experience_base = min(reviewer.reviews_completed / 100, 1.0)  # Normalize to 0-1
        
        # Quality rating factor
        quality_factor = reviewer.quality_rating
        
        # Role-based experience bonus
        role_bonus = {
            ReviewerRole.GENERAL_PROOFREADER: 0.0,
            ReviewerRole.SUBJECT_MATTER_EXPERT: 0.2,
            ReviewerRole.ACADEMIC_CONSULTANT: 0.4,
            ReviewerRole.SENIOR_REVIEWER: 0.3
        }.get(reviewer.role, 0.0)
        
        # Complexity alignment
        if complexity_rating.overall_complexity > 0.8 and reviewer.role == ReviewerRole.GENERAL_PROOFREADER:
            complexity_penalty = -0.3  # GP may struggle with very complex content
        elif complexity_rating.overall_complexity < 0.3 and reviewer.role == ReviewerRole.ACADEMIC_CONSULTANT:
            complexity_penalty = -0.1  # Overqualified but not heavily penalized
        else:
            complexity_penalty = 0.0
        
        return max(experience_base + quality_factor * 0.3 + role_bonus + complexity_penalty, 0.0)
    
    def _calculate_workload_feasibility(self, reviewer: ReviewerProfile, priority: str) -> float:
        """Calculate workload feasibility score."""
        # Current workload factor
        workload_ratio = reviewer.current_review_load / reviewer.max_concurrent_reviews
        workload_score = 1.0 - workload_ratio
        
        # Average review time factor
        if reviewer.average_review_time_hours > 0:
            time_efficiency = max(1.0 - (reviewer.average_review_time_hours / 48), 0.2)  # 48h as reasonable max
        else:
            time_efficiency = 0.8  # Default for new reviewers
        
        # Priority adjustment
        priority_bonus = {
            'critical': 0.2,
            'high': 0.1,
            'standard': 0.0,
            'low': -0.1
        }.get(priority, 0.0)
        
        return max(workload_score * 0.7 + time_efficiency * 0.3 + priority_bonus, 0.0)
    
    def _calculate_specialization_bonus(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> float:
        """Calculate bonus for relevant specializations."""
        bonus = 0.0
        
        # Check primary expertise match
        if reviewer.primary_expertise:
            if 'sanskrit' in reviewer.primary_expertise.lower() and complexity_rating.sanskrit_term_count > 5:
                bonus += 0.3
            elif 'academic' in reviewer.primary_expertise.lower() and complexity_rating.academic_reference_count > 2:
                bonus += 0.2
            elif 'vedanta' in reviewer.primary_expertise.lower() and complexity_rating.theological_concept_count > 3:
                bonus += 0.25
        
        # Check secondary expertise
        for secondary in reviewer.secondary_expertise:
            if 'iast' in secondary.lower() and complexity_rating.dimension_scores.get(ComplexityDimension.TECHNICAL_TERMINOLOGY, 0) > 0.6:
                bonus += 0.15
            elif 'citation' in secondary.lower() and complexity_rating.dimension_scores.get(ComplexityDimension.CITATION_COMPLEXITY, 0) > 0.5:
                bonus += 0.1
        
        return min(bonus, 0.5)  # Cap bonus at 0.5
    
    def _calculate_context_awareness(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> float:
        """Calculate Epic 4.1 MCP context awareness score."""
        # Base context awareness from MCP analysis
        base_score = complexity_rating.mcp_context_analysis.get('contextual_coherence', 0.5)
        
        # Reviewer's cultural context experience
        cultural_experience = 0.5  # Default - would be enhanced with reviewer cultural background data
        
        # Role-based context awareness
        role_awareness = {
            ReviewerRole.GENERAL_PROOFREADER: 0.3,
            ReviewerRole.SUBJECT_MATTER_EXPERT: 0.7,
            ReviewerRole.ACADEMIC_CONSULTANT: 0.9,
            ReviewerRole.SENIOR_REVIEWER: 0.8
        }.get(reviewer.role, 0.5)
        
        return (base_score * 0.4 + cultural_experience * 0.3 + role_awareness * 0.3)
    
    def _calculate_semantic_compatibility(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> float:
        """Calculate semantic compatibility using MCP analysis."""
        semantic_complexity = complexity_rating.mcp_context_analysis.get('semantic_complexity', 0.5)
        
        # Reviewer's semantic processing capability (estimated from role and experience)
        reviewer_capability = min(
            reviewer.quality_rating * 0.5 +
            (reviewer.reviews_completed / 50) * 0.3 +
            {'novice': 0.2, 'intermediate': 0.5, 'expert': 0.8, 'master': 1.0}.get(
                getattr(reviewer.skills[0], 'certification_level', 'intermediate') if reviewer.skills else 'intermediate', 0.5
            ) * 0.2,
            1.0
        )
        
        # Compatibility is how well capability matches requirement
        if semantic_complexity <= reviewer_capability:
            return 1.0 - (reviewer_capability - semantic_complexity) * 0.3  # Small penalty for overqualification
        else:
            return max(reviewer_capability / semantic_complexity, 0.2)  # Larger penalty for underqualification
    
    def _calculate_cultural_context_match(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> float:
        """Calculate cultural context match score."""
        cultural_depth = complexity_rating.mcp_context_analysis.get('cultural_context_depth', 0.5)
        
        # Estimate reviewer's cultural context knowledge
        # This would be enhanced with actual cultural background data
        cultural_knowledge = 0.5  # Default baseline
        
        # Role-based cultural knowledge
        if reviewer.role in [ReviewerRole.SUBJECT_MATTER_EXPERT, ReviewerRole.ACADEMIC_CONSULTANT]:
            cultural_knowledge += 0.3
        
        # Experience bonus
        if reviewer.reviews_completed > 20:
            cultural_knowledge += 0.2
        
        return min(cultural_knowledge / max(cultural_depth, 0.1), 1.0)
    
    def _calculate_academic_qualification_match(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> float:
        """Calculate Epic 4.5 academic qualification match."""
        if not complexity_rating.academic_validation_required:
            return 1.0  # No special requirements
        
        # Academic credentials score
        credentials_score = min(len(reviewer.academic_credentials) / 3, 1.0)  # 3+ credentials is excellent
        
        # Publication history score
        publications_score = min(len(reviewer.publication_history) / 5, 1.0)  # 5+ publications is excellent
        
        # Institutional affiliation bonus
        affiliation_bonus = 0.2 if reviewer.institutional_affiliation else 0.0
        
        # Role-based academic capability
        role_capability = {
            ReviewerRole.GENERAL_PROOFREADER: 0.2,
            ReviewerRole.SUBJECT_MATTER_EXPERT: 0.6,
            ReviewerRole.ACADEMIC_CONSULTANT: 1.0,
            ReviewerRole.SENIOR_REVIEWER: 0.8
        }.get(reviewer.role, 0.3)
        
        return min(
            credentials_score * 0.3 + 
            publications_score * 0.3 + 
            role_capability * 0.4 + 
            affiliation_bonus, 
            1.0
        )
    
    def _determine_confidence_level(self, overall_score: float) -> MatchingConfidence:
        """Determine confidence level based on overall match score."""
        if overall_score >= self.matching_config['excellent_match_threshold']:
            return MatchingConfidence.EXCELLENT
        elif overall_score >= self.matching_config['good_match_threshold']:
            return MatchingConfidence.GOOD
        elif overall_score >= 0.5:
            return MatchingConfidence.MODERATE
        elif overall_score >= self.matching_config['min_match_score_threshold']:
            return MatchingConfidence.POOR
        else:
            return MatchingConfidence.UNFIT
    
    def _estimate_review_time(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> float:
        """Estimate review time in hours."""
        # Base time from reviewer's history
        base_time = reviewer.average_review_time_hours if reviewer.average_review_time_hours > 0 else 8.0
        
        # Complexity multiplier
        complexity_multiplier = 0.5 + complexity_rating.overall_complexity * 1.5
        
        # Role efficiency factor
        role_efficiency = {
            ReviewerRole.GENERAL_PROOFREADER: 1.0,
            ReviewerRole.SUBJECT_MATTER_EXPERT: 1.2,  # More thorough review
            ReviewerRole.ACADEMIC_CONSULTANT: 1.5,   # Most thorough
            ReviewerRole.SENIOR_REVIEWER: 1.1
        }.get(reviewer.role, 1.0)
        
        return base_time * complexity_multiplier * role_efficiency
    
    def _assess_potential_issues(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> List[str]:
        """Assess potential issues with this reviewer-content match."""
        issues = []
        
        # Complexity vs experience mismatch
        if complexity_rating.overall_complexity > 0.8 and reviewer.role == ReviewerRole.GENERAL_PROOFREADER:
            issues.append("Content complexity may exceed GP reviewer capabilities")
        
        # Workload concerns
        if reviewer.current_review_load >= reviewer.max_concurrent_reviews * 0.8:
            issues.append("Reviewer approaching maximum workload capacity")
        
        # Academic requirements vs qualifications
        if complexity_rating.academic_validation_required and len(reviewer.academic_credentials) == 0:
            issues.append("Academic validation required but reviewer lacks formal credentials")
        
        # Sanskrit content vs expertise
        if complexity_rating.sanskrit_term_count > 10 and not any('sanskrit' in skill.skill_area.lower() for skill in reviewer.skills):
            issues.append("High Sanskrit content but no documented Sanskrit expertise")
        
        return issues
    
    def _assess_quality_risk(self, overall_score: float, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> str:
        """Assess quality risk level for this match."""
        if overall_score >= 0.8:
            return "low"
        elif overall_score >= 0.6:
            return "medium"
        elif overall_score >= 0.4:
            return "high"
        else:
            return "very_high"
    
    def _calculate_escalation_likelihood(self, complexity_rating: ComplexityRating, reviewer: ReviewerProfile) -> float:
        """Calculate likelihood of consultant escalation."""
        # Base likelihood from complexity
        base_likelihood = complexity_rating.overall_complexity
        
        # Increase likelihood if academic validation required
        if complexity_rating.academic_validation_required:
            base_likelihood += 0.2
        
        # Reviewer role factor
        if reviewer.role == ReviewerRole.GENERAL_PROOFREADER:
            base_likelihood += 0.3  # GPs more likely to escalate
        elif reviewer.role == ReviewerRole.ACADEMIC_CONSULTANT:
            base_likelihood -= 0.4  # Consultants less likely to escalate
        
        return min(base_likelihood, 1.0)
    
    def _check_circuit_breaker(self) -> bool:
        """Check Epic 4.1 circuit breaker status."""
        if self.circuit_breaker_open:
            if time.time() > self.circuit_breaker_reset_time:
                self.circuit_breaker_open = False
                self.circuit_breaker_failures = 0
                self.logger.info("Circuit breaker reset - resuming MCP analysis")
                return False
            return True
        return False
    
    def _create_fallback_complexity_rating(self, content_id: str, segments: List[SRTSegment]) -> ComplexityRating:
        """Create fallback complexity rating when MCP analysis fails."""
        combined_text = " ".join([segment.text for segment in segments])
        basic_analysis = self._perform_basic_complexity_analysis(combined_text)
        fallback_mcp = self._fallback_complexity_analysis(combined_text)
        
        # Simple dimension scores based on basic analysis
        dimension_scores = {
            ComplexityDimension.LINGUISTIC_COMPLEXITY: min(basic_analysis['unique_vocabulary_ratio'] * 2, 1.0),
            ComplexityDimension.SANSKRIT_CONTENT_DENSITY: min(basic_analysis['sanskrit_terms'] / max(basic_analysis['total_words'] / 20, 1), 1.0),
            ComplexityDimension.ACADEMIC_RIGOR_REQUIRED: min(basic_analysis['academic_references'] / max(basic_analysis['total_words'] / 50, 1), 1.0),
            ComplexityDimension.THEOLOGICAL_DEPTH: min(basic_analysis['theological_concepts'] / max(basic_analysis['total_words'] / 30, 1), 1.0),
            ComplexityDimension.TECHNICAL_TERMINOLOGY: 0.5,  # Default
            ComplexityDimension.CITATION_COMPLEXITY: min(basic_analysis['academic_references'] / max(basic_analysis['total_words'] / 100, 1), 1.0)
        }
        
        overall_complexity = self._calculate_overall_complexity(dimension_scores)
        
        rating = ComplexityRating(
            content_id=content_id,
            overall_complexity=overall_complexity,
            dimension_scores=dimension_scores,
            sanskrit_term_count=basic_analysis['sanskrit_terms'],
            theological_concept_count=basic_analysis['theological_concepts'],
            academic_reference_count=basic_analysis['academic_references'],
            unique_vocabulary_ratio=basic_analysis['unique_vocabulary_ratio'],
            mcp_context_analysis=fallback_mcp,
            semantic_complexity_score=fallback_mcp['semantic_complexity'],
            contextual_coherence_score=fallback_mcp['contextual_coherence'],
            analysis_confidence=0.6,  # Lower confidence for fallback
            circuit_breaker_used=True
        )
        
        self._assess_academic_requirements(rating)
        return rating
    
    def get_matching_statistics(self) -> Dict[str, Any]:
        """Get comprehensive matching system statistics."""
        with self.lock:
            success_rate = statistics.mean(self.match_success_rates) if self.match_success_rates else 0.0
            avg_analysis_time = statistics.mean(self.analysis_times) if self.analysis_times else 0.0
            
            return {
                'performance': {
                    'complexity_analyses_completed': self.matching_statistics['complexity_analyses'],
                    'matching_attempts': self.matching_statistics['matching_attempts'],
                    'successful_matches': self.matching_statistics['successful_matches'],
                    'match_success_rate': success_rate,
                    'average_analysis_time_ms': avg_analysis_time
                },
                'epic_4_1_integration': {
                    'mcp_analysis_enabled': True,
                    'circuit_breaker_status': 'open' if self.circuit_breaker_open else 'closed',
                    'circuit_breaker_failures': self.circuit_breaker_failures,
                    'cached_complexity_ratings': len(self.complexity_cache)
                },
                'system_health': {
                    'mcp_client_operational': self.mcp_client_manager is not None,
                    'performance_monitoring_active': True,
                    'telemetry_collection_active': True
                }
            }