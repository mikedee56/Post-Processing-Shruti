"""
Context-Dependent Spelling Normalization System

This module provides spelling normalization capabilities that consider context
for shortened word expansion, formal spelling standardization, and consistency
checking across document segments.
"""

import re
import json
import yaml
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from collections import defaultdict, Counter

from utils.logger_config import get_logger


class NormalizationType(Enum):
    """Types of spelling normalization."""
    EXPANSION = "expansion"
    STANDARDIZATION = "standardization"
    VARIANT_DETECTION = "variant_detection"
    CONSISTENCY_CHECK = "consistency_check"


@dataclass
class SpellingVariant:
    """Represents a spelling variant."""
    original: str
    canonical: str
    variant_type: str
    context_score: float
    frequency: int = 0
    positions: List[int] = field(default_factory=list)


@dataclass
class NormalizationRule:
    """Spelling normalization rule."""
    id: str
    pattern: str
    replacement: str
    rule_type: NormalizationType
    context_required: List[str] = field(default_factory=list)
    confidence: float = 1.0
    enabled: bool = True
    description: str = ""


@dataclass
class NormalizationResult:
    """Result of spelling normalization."""
    original_text: str
    normalized_text: str
    changes_made: List[Tuple[str, str, int]]  # (original, replacement, position)
    confidence_score: float
    normalization_types: List[NormalizationType]
    consistency_issues: List[str] = field(default_factory=list)


class SpellingNormalizer:
    """
    Context-aware spelling normalization system.
    
    Handles shortened word expansion, formal spelling standardization,
    variant detection, and consistency checking for Sanskrit/Hindi terms.
    """

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize spelling normalizer.
        
        Args:
            config_file: Path to normalization rules configuration
        """
        self.logger = get_logger(__name__)
        self.normalization_rules: Dict[str, NormalizationRule] = {}
        self.expansion_mappings: Dict[str, str] = {}
        self.variant_groups: Dict[str, List[str]] = {}
        self.document_consistency: Dict[str, Dict[str, int]] = {}
        
        # Load configuration
        if config_file and config_file.exists():
            self.load_configuration(config_file)
        else:
            self._initialize_default_rules()
        
        self.logger.info(f"SpellingNormalizer initialized with {len(self.normalization_rules)} rules")

    def _initialize_default_rules(self) -> None:
        """Initialize default normalization rules."""
        # Shortened word expansion rules
        self._add_expansion_rules()
        
        # Standardization rules
        self._add_standardization_rules()
        
        # Variant detection rules
        self._add_variant_detection_rules()
        
        # Initialize variant groups
        self._initialize_variant_groups()

    def _add_expansion_rules(self) -> None:
        """Add shortened word expansion rules."""
        expansion_rules = [
            {
                'id': 'sanskrit_shortenings',
                'pattern': r'\b(gita|geeta)\b',
                'replacement': 'Bhagavad Gita',
                'rule_type': NormalizationType.EXPANSION,
                'context_required': ['bhagavad', 'scripture', 'verse', 'chapter'],
                'confidence': 0.8,
                'description': 'Expand Gita to Bhagavad Gita when in scripture context'
            },
            {
                'id': 'yoga_expansion',
                'pattern': r'\byog\b',
                'replacement': 'yoga',
                'rule_type': NormalizationType.EXPANSION,
                'context_required': ['practice', 'path', 'karma', 'bhakti', 'raja'],
                'confidence': 0.9,
                'description': 'Expand yog to yoga in practice context'
            },
            {
                'id': 'dharma_expansion',
                'pattern': r'\bdhrm\b',
                'replacement': 'dharma',
                'rule_type': NormalizationType.EXPANSION,
                'context_required': ['duty', 'righteous', 'law', 'virtue'],
                'confidence': 0.85,
                'description': 'Expand dhrm to dharma'
            },
            {
                'id': 'krishna_expansion',
                'pattern': r'\bkrsna\b',
                'replacement': 'Krishna',
                'rule_type': NormalizationType.EXPANSION,
                'context_required': ['lord', 'deity', 'god', 'avatar'],
                'confidence': 0.9,
                'description': 'Expand krsna to Krishna'
            }
        ]
        
        for rule_data in expansion_rules:
            rule = NormalizationRule(**rule_data)
            self.normalization_rules[rule.id] = rule
            # Also add to expansion mappings for quick lookup
            if rule.rule_type == NormalizationType.EXPANSION:
                pattern_word = re.sub(r'\\b|\(|\)|\\', '', rule.pattern)
                self.expansion_mappings[pattern_word] = rule.replacement

    def _add_standardization_rules(self) -> None:
        """Add formal spelling standardization rules."""
        standardization_rules = [
            {
                'id': 'deity_capitalization',
                'pattern': r'\b(krishna|arjuna|rama|shiva|hanuman)\b',
                'replacement': lambda m: m.group(1).title(),
                'rule_type': NormalizationType.STANDARDIZATION,
                'confidence': 0.95,
                'description': 'Capitalize deity names'
            },
            {
                'id': 'scripture_capitalization',
                'pattern': r'\b(bhagavad gita|upanishad|vedas?|puranas?)\b',
                'replacement': lambda m: m.group(1).title(),
                'rule_type': NormalizationType.STANDARDIZATION,
                'confidence': 0.95,
                'description': 'Capitalize scripture names'
            },
            {
                'id': 'concept_standardization',
                'pattern': r'\b(dharama|dhrama)\b',
                'replacement': 'dharma',
                'rule_type': NormalizationType.STANDARDIZATION,
                'confidence': 0.9,
                'description': 'Standardize dharma spelling variants'
            },
            {
                'id': 'practice_standardization',
                'pattern': r'\b(yogaa|yogh)\b',
                'replacement': 'yoga',
                'rule_type': NormalizationType.STANDARDIZATION,
                'confidence': 0.9,
                'description': 'Standardize yoga spelling variants'
            }
        ]
        
        for rule_data in standardization_rules:
            rule = NormalizationRule(**rule_data)
            self.normalization_rules[rule.id] = rule

    def _add_variant_detection_rules(self) -> None:
        """Add spelling variant detection rules."""
        variant_rules = [
            {
                'id': 'long_vowel_variants',
                'pattern': r'\b(\w*)(aa|ii|uu|oo|ee)(\w*)\b',
                'replacement': r'\1\2\3',
                'rule_type': NormalizationType.VARIANT_DETECTION,
                'confidence': 0.7,
                'description': 'Detect long vowel variants'
            },
            {
                'id': 'aspiration_variants',
                'pattern': r'\b(\w*)(kh|gh|ch|jh|th|dh|ph|bh)(\w*)\b',
                'replacement': r'\1\2\3',
                'rule_type': NormalizationType.VARIANT_DETECTION,
                'confidence': 0.8,
                'description': 'Detect aspiration variants'
            },
            {
                'id': 'retroflex_variants',
                'pattern': r'\b(\w*)(ṭ|ḍ|ṇ|ṛ|ṣ)(\w*)\b',
                'replacement': r'\1\2\3',
                'rule_type': NormalizationType.VARIANT_DETECTION,
                'confidence': 0.75,
                'description': 'Detect retroflex variants'
            }
        ]
        
        for rule_data in variant_rules:
            rule = NormalizationRule(**rule_data)
            self.normalization_rules[rule.id] = rule

    def _initialize_variant_groups(self) -> None:
        """Initialize spelling variant groups."""
        self.variant_groups = {
            'dharma': ['dharma', 'dharama', 'dhrama', 'dhrma'],
            'yoga': ['yoga', 'yog', 'yogaa', 'yogh'],
            'krishna': ['krishna', 'krsna', 'krshna', 'krisna'],
            'arjuna': ['arjuna', 'arjun', 'arjuuna'],
            'karma': ['karma', 'karm', 'karmaa'],
            'bhakti': ['bhakti', 'bhakti', 'bhakthi'],
            'gita': ['gita', 'geeta', 'geet'],
            'upanishad': ['upanishad', 'upanishads', 'upanisad'],
            'vedanta': ['vedanta', 'vedant', 'vedaanta'],
            'meditation': ['meditation', 'dhyana', 'dhyan']
        }

    def normalize_text(self, text: str, context: List[str] = None, 
                      document_id: str = None) -> NormalizationResult:
        """
        Normalize spelling in text with context awareness.
        
        Args:
            text: Text to normalize
            context: Surrounding context words
            document_id: Document identifier for consistency tracking
            
        Returns:
            NormalizationResult with normalization details
        """
        context = context or []
        context_text = " ".join(context).lower()
        
        original_text = text
        normalized_text = text
        changes_made = []
        normalization_types = []
        consistency_issues = []
        
        # Track document consistency
        if document_id:
            if document_id not in self.document_consistency:
                self.document_consistency[document_id] = defaultdict(int)
        
        # Apply expansion rules
        normalized_text, expansion_changes = self._apply_expansion_rules(
            normalized_text, context_text
        )
        changes_made.extend(expansion_changes)
        if expansion_changes:
            normalization_types.append(NormalizationType.EXPANSION)
        
        # Apply standardization rules
        normalized_text, standard_changes = self._apply_standardization_rules(
            normalized_text, context_text
        )
        changes_made.extend(standard_changes)
        if standard_changes:
            normalization_types.append(NormalizationType.STANDARDIZATION)
        
        # Detect and handle variants
        variants = self._detect_spelling_variants(normalized_text)
        normalized_text, variant_changes = self._handle_spelling_variants(
            normalized_text, variants, context_text
        )
        changes_made.extend(variant_changes)
        if variants:
            normalization_types.append(NormalizationType.VARIANT_DETECTION)
        
        # Check consistency across document
        if document_id:
            doc_consistency_issues = self._check_document_consistency(
                normalized_text, document_id
            )
            consistency_issues.extend(doc_consistency_issues)
            if doc_consistency_issues:
                normalization_types.append(NormalizationType.CONSISTENCY_CHECK)
        
        # Calculate overall confidence
        confidence_score = self._calculate_normalization_confidence(
            changes_made, consistency_issues, context
        )
        
        return NormalizationResult(
            original_text=original_text,
            normalized_text=normalized_text,
            changes_made=changes_made,
            confidence_score=confidence_score,
            normalization_types=normalization_types,
            consistency_issues=consistency_issues
        )

    def _apply_expansion_rules(self, text: str, context: str) -> Tuple[str, List[Tuple[str, str, int]]]:
        """Apply shortened word expansion rules."""
        changes = []
        normalized_text = text
        
        for rule_id, rule in self.normalization_rules.items():
            if rule.rule_type != NormalizationType.EXPANSION or not rule.enabled:
                continue
            
            # Check if context is required and present
            if rule.context_required:
                context_present = any(
                    req_context in context for req_context in rule.context_required
                )
                if not context_present:
                    continue
            
            # Apply the expansion
            pattern = re.compile(rule.pattern, re.IGNORECASE)
            matches = list(pattern.finditer(normalized_text))
            
            for match in reversed(matches):  # Reverse to maintain positions
                original = match.group(0)
                replacement = rule.replacement
                if callable(replacement):
                    replacement = replacement(match)
                
                changes.append((original, replacement, match.start()))
                normalized_text = (
                    normalized_text[:match.start()] + 
                    replacement + 
                    normalized_text[match.end():]
                )
        
        return normalized_text, changes

    def _apply_standardization_rules(self, text: str, context: str) -> Tuple[str, List[Tuple[str, str, int]]]:
        """Apply formal spelling standardization rules."""
        changes = []
        normalized_text = text
        
        for rule_id, rule in self.normalization_rules.items():
            if rule.rule_type != NormalizationType.STANDARDIZATION or not rule.enabled:
                continue
            
            pattern = re.compile(rule.pattern, re.IGNORECASE)
            matches = list(pattern.finditer(normalized_text))
            
            for match in reversed(matches):  # Reverse to maintain positions
                original = match.group(0)
                replacement = rule.replacement
                if callable(replacement):
                    replacement = replacement(match)
                
                changes.append((original, replacement, match.start()))
                normalized_text = (
                    normalized_text[:match.start()] + 
                    replacement + 
                    normalized_text[match.end():]
                )
        
        return normalized_text, changes

    def _detect_spelling_variants(self, text: str) -> List[SpellingVariant]:
        """Detect spelling variants in text."""
        variants = []
        words = re.findall(r'\b\w+\b', text.lower())
        word_positions = [(m.start(), m.group()) for m in re.finditer(r'\b\w+\b', text.lower())]
        
        # Check against variant groups
        for canonical, variant_list in self.variant_groups.items():
            for word, (pos, _) in zip(words, word_positions):
                if word in variant_list and word != canonical:
                    variant = SpellingVariant(
                        original=word,
                        canonical=canonical,
                        variant_type='known_variant',
                        context_score=0.8,
                        frequency=1,
                        positions=[pos]
                    )
                    variants.append(variant)
        
        return variants

    def _handle_spelling_variants(self, text: str, variants: List[SpellingVariant], 
                                context: str) -> Tuple[str, List[Tuple[str, str, int]]]:
        """Handle detected spelling variants."""
        changes = []
        normalized_text = text
        
        # Sort variants by position (descending) to maintain positions during replacement
        variants.sort(key=lambda v: v.positions[0] if v.positions else 0, reverse=True)
        
        for variant in variants:
            if not variant.positions:
                continue
            
            # Check context relevance
            context_relevant = self._is_variant_context_relevant(
                variant, context
            )
            
            if context_relevant and variant.context_score > 0.5:
                for pos in variant.positions:
                    # Find the word at this position
                    pattern = rf'\b{re.escape(variant.original)}\b'
                    match = re.search(pattern, normalized_text[pos:pos+20], re.IGNORECASE)
                    
                    if match:
                        actual_pos = pos + match.start()
                        changes.append((variant.original, variant.canonical, actual_pos))
                        
                        # Replace the variant
                        normalized_text = (
                            normalized_text[:actual_pos] + 
                            variant.canonical + 
                            normalized_text[actual_pos + len(variant.original):]
                        )
        
        return normalized_text, changes

    def _is_variant_context_relevant(self, variant: SpellingVariant, context: str) -> bool:
        """Check if variant replacement is contextually relevant."""
        # Define context relevance for common terms
        context_mappings = {
            'dharma': ['duty', 'righteousness', 'virtue', 'law', 'moral'],
            'yoga': ['practice', 'union', 'path', 'discipline', 'meditation'],
            'krishna': ['lord', 'deity', 'god', 'avatar', 'divine'],
            'karma': ['action', 'deed', 'work', 'consequence'],
            'gita': ['scripture', 'text', 'verse', 'chapter', 'teaching']
        }
        
        if variant.canonical in context_mappings:
            relevant_contexts = context_mappings[variant.canonical]
            return any(ctx in context.lower() for ctx in relevant_contexts)
        
        return True  # Default to relevant if no specific mapping

    def _check_document_consistency(self, text: str, document_id: str) -> List[str]:
        """Check consistency across document segments."""
        issues = []
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Update document word counts
        for word in words:
            self.document_consistency[document_id][word] += 1
        
        # Check for inconsistent usage of variants
        doc_words = self.document_consistency[document_id]
        
        for canonical, variants in self.variant_groups.items():
            used_variants = []
            for variant in variants:
                if variant in doc_words and doc_words[variant] > 0:
                    used_variants.append((variant, doc_words[variant]))
            
            # If multiple variants are used, flag as consistency issue
            if len(used_variants) > 1:
                sorted_variants = sorted(used_variants, key=lambda x: x[1], reverse=True)
                primary_variant = sorted_variants[0][0]
                
                for variant, count in sorted_variants[1:]:
                    if count > 1:  # Only flag if used more than once
                        issues.append(
                            f"Inconsistent usage: '{variant}' used {count} times, "
                            f"consider standardizing to '{primary_variant}'"
                        )
        
        return issues

    def _calculate_normalization_confidence(self, changes: List[Tuple[str, str, int]], 
                                          consistency_issues: List[str], 
                                          context: List[str]) -> float:
        """Calculate overall confidence score for normalization."""
        base_confidence = 0.8
        
        # Boost for context presence
        if context and len(context) > 2:
            base_confidence += 0.1
        
        # Reduce for many changes (might indicate errors)
        if len(changes) > 5:
            base_confidence -= 0.1
        
        # Reduce for consistency issues
        if consistency_issues:
            base_confidence -= 0.05 * len(consistency_issues)
        
        # Boost for successful normalizations
        if changes:
            base_confidence += 0.05
        
        return max(0.0, min(1.0, base_confidence))

    def expand_shortened_words(self, text: str, context: List[str] = None) -> str:
        """
        Expand shortened words based on context.
        
        Args:
            text: Text with potential shortened words
            context: Context for expansion decisions
            
        Returns:
            Text with expanded words
        """
        context_str = " ".join(context or []).lower()
        expanded_text = text
        
        for shortened, expanded in self.expansion_mappings.items():
            # Simple context check
            if shortened.lower() in text.lower():
                # Check if expansion is contextually appropriate
                pattern = rf'\b{re.escape(shortened)}\b'
                if re.search(pattern, text, re.IGNORECASE):
                    expanded_text = re.sub(pattern, expanded, expanded_text, flags=re.IGNORECASE)
        
        return expanded_text

    def get_spelling_suggestions(self, word: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Get spelling suggestions for a word.
        
        Args:
            word: Word to get suggestions for
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of (suggestion, confidence) tuples
        """
        suggestions = []
        word_lower = word.lower()
        
        # Check variant groups
        for canonical, variants in self.variant_groups.items():
            if word_lower in variants:
                suggestions.append((canonical, 0.9))
                break
        
        # Check expansion mappings
        if word_lower in self.expansion_mappings:
            suggestions.append((self.expansion_mappings[word_lower], 0.85))
        
        # Simple edit distance suggestions (basic implementation)
        for canonical in self.variant_groups.keys():
            if canonical not in [s[0] for s in suggestions]:
                if self._simple_edit_distance(word_lower, canonical) <= 2:
                    confidence = 1.0 - (self._simple_edit_distance(word_lower, canonical) * 0.2)
                    suggestions.append((canonical, confidence))
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]

    def _simple_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate simple edit distance between two strings."""
        if len(s1) < len(s2):
            return self._simple_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def load_configuration(self, config_file: Path) -> bool:
        """
        Load normalization configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if successful
        """
        try:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            else:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            
            # Load rules
            if 'normalization_rules' in config_data:
                for rule_data in config_data['normalization_rules']:
                    rule = NormalizationRule(**rule_data)
                    self.normalization_rules[rule.id] = rule
            
            # Load expansion mappings
            if 'expansion_mappings' in config_data:
                self.expansion_mappings.update(config_data['expansion_mappings'])
            
            # Load variant groups
            if 'variant_groups' in config_data:
                self.variant_groups.update(config_data['variant_groups'])
            
            self.logger.info(f"Loaded normalization configuration from {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False

    def get_normalization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive normalization statistics."""
        rule_types = defaultdict(int)
        for rule in self.normalization_rules.values():
            rule_types[rule.rule_type.value] += 1
        
        return {
            'total_rules': len(self.normalization_rules),
            'expansion_mappings': len(self.expansion_mappings),
            'variant_groups': len(self.variant_groups),
            'rule_types': dict(rule_types),
            'documents_tracked': len(self.document_consistency),
            'total_variants': sum(len(variants) for variants in self.variant_groups.values())
        }