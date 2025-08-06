"""
Contextual Rule Engine for Sanskrit/Hindi Text Processing

This module provides rule-based contextual corrections and consistency validation
for Sanskrit/Hindi terms, including compound term detection and standardization.
"""

import re
import yaml
import json
from typing import Dict, List, Tuple, Optional, Set, Any, Pattern
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

from utils.logger_config import get_logger


class RuleType(Enum):
    """Types of contextual rules."""
    COMPOUND_TERM = "compound_term"
    SEQUENTIAL_DEPENDENCY = "sequential_dependency"
    CATEGORY_CONSISTENCY = "category_consistency"
    CONTEXTUAL_SUBSTITUTION = "contextual_substitution"
    PROXIMITY_RULE = "proximity_rule"


class RulePriority(Enum):
    """Rule execution priorities."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class ContextualRule:
    """Represents a contextual correction rule."""
    id: str
    name: str
    rule_type: RuleType
    priority: RulePriority
    pattern: str
    replacement: str
    context_patterns: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    confidence_boost: float = 0.0
    enabled: bool = True
    description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class RuleMatch:
    """Result of rule matching."""
    rule_id: str
    rule_name: str
    original_text: str
    corrected_text: str
    match_position: int
    match_length: int
    confidence: float
    context_matched: List[str]
    rule_type: RuleType


@dataclass
class CompoundTerm:
    """Represents a compound Sanskrit/Hindi term."""
    term: str
    components: List[str]
    canonical_form: str
    transliteration: str
    category: str
    confidence: float = 1.0
    variations: List[str] = field(default_factory=list)


class ContextualRuleEngine:
    """
    Rule-based engine for contextual Sanskrit/Hindi corrections.
    
    Handles compound term detection, sequential dependencies, category
    consistency, and configurable contextual substitution rules.
    """

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize contextual rule engine.
        
        Args:
            config_file: Path to rules configuration file
        """
        self.logger = get_logger(__name__)
        self.rules: Dict[str, ContextualRule] = {}
        self.compiled_patterns: Dict[str, Pattern] = {}
        self.compound_terms: Dict[str, CompoundTerm] = {}
        self.category_dependencies: Dict[str, List[str]] = {}
        
        # Load configuration
        self.config_file = config_file
        if config_file and config_file.exists():
            self.load_rules_from_config(config_file)
        else:
            self._initialize_default_rules()
        
        self.logger.info(f"ContextualRuleEngine initialized with {len(self.rules)} rules")

    def _initialize_default_rules(self) -> None:
        """Initialize default contextual rules."""
        # Compound term rules
        self._add_compound_term_rules()
        
        # Sequential dependency rules
        self._add_sequential_dependency_rules()
        
        # Category consistency rules
        self._add_category_consistency_rules()
        
        # Contextual substitution rules
        self._add_contextual_substitution_rules()
        
        # Compile all patterns
        self._compile_rule_patterns()

    def _add_compound_term_rules(self) -> None:
        """Add compound term detection and standardization rules."""
        compound_rules = [
            {
                'id': 'karma_yoga',
                'name': 'Karma Yoga Standardization',
                'rule_type': RuleType.COMPOUND_TERM,
                'priority': RulePriority.HIGH,
                'pattern': r'\b(karma)\s+(yog[a]?)\b',
                'replacement': 'karma yoga',
                'context_patterns': ['practice', 'path', 'teaching', 'philosophy'],
                'confidence_boost': 0.15,
                'description': 'Standardize karma yoga compound term',
                'examples': ['karma yog -> karma yoga', 'karm yoga -> karma yoga']
            },
            {
                'id': 'bhakti_yoga',
                'name': 'Bhakti Yoga Standardization',
                'rule_type': RuleType.COMPOUND_TERM,
                'priority': RulePriority.HIGH,
                'pattern': r'\b(bhakti)\s+(yog[a]?)\b',
                'replacement': 'bhakti yoga',
                'context_patterns': ['devotion', 'love', 'surrender', 'practice'],
                'confidence_boost': 0.15
            },
            {
                'id': 'raja_yoga',
                'name': 'Raja Yoga Standardization',
                'rule_type': RuleType.COMPOUND_TERM,
                'priority': RulePriority.HIGH,
                'pattern': r'\b(raj[a]?)\s+(yog[a]?)\b',
                'replacement': 'raja yoga',
                'context_patterns': ['meditation', 'control', 'royal', 'practice'],
                'confidence_boost': 0.15
            },
            {
                'id': 'bhagavad_gita',
                'name': 'Bhagavad Gita Standardization',
                'rule_type': RuleType.COMPOUND_TERM,
                'priority': RulePriority.HIGH,
                'pattern': r'\b(bhagavad?|bhagwad)\s+(git[a]?|geet[a]?)\b',
                'replacement': 'Bhagavad Gita',
                'context_patterns': ['scripture', 'text', 'verse', 'chapter'],
                'confidence_boost': 0.2
            }
        ]
        
        for rule_data in compound_rules:
            rule = ContextualRule(**rule_data)
            self.rules[rule.id] = rule

    def _add_sequential_dependency_rules(self) -> None:
        """Add sequential dependency validation rules."""
        sequential_rules = [
            {
                'id': 'chapter_verse_sequence',
                'name': 'Chapter-Verse Sequential Validation',
                'rule_type': RuleType.SEQUENTIAL_DEPENDENCY,
                'priority': RulePriority.MEDIUM,
                'pattern': r'\b(chapter)\s+(\d+)\s+(verse)\s+(\d+)\b',
                'replacement': r'chapter \2 verse \4',
                'conditions': {'validate_sequence': True},
                'confidence_boost': 0.1
            },
            {
                'id': 'sanskrit_term_sequence',
                'name': 'Sanskrit Term Sequential Context',
                'rule_type': RuleType.SEQUENTIAL_DEPENDENCY,
                'priority': RulePriority.MEDIUM,
                'pattern': r'\b(dharma)\s+(and|or)\s+(karma)\b',
                'replacement': r'dharma and karma',
                'context_patterns': ['concept', 'teaching', 'philosophy'],
                'confidence_boost': 0.1
            }
        ]
        
        for rule_data in sequential_rules:
            rule = ContextualRule(**rule_data)
            self.rules[rule.id] = rule

    def _add_category_consistency_rules(self) -> None:
        """Add category consistency validation rules."""
        # Define category dependencies
        self.category_dependencies = {
            'scripture': ['verse', 'chapter', 'text'],
            'deity': ['worship', 'devotion', 'prayer'],
            'practice': ['meditation', 'yoga', 'discipline'],
            'philosophy': ['concept', 'principle', 'teaching']
        }
        
        consistency_rules = [
            {
                'id': 'deity_context_consistency',
                'name': 'Deity Context Consistency',
                'rule_type': RuleType.CATEGORY_CONSISTENCY,
                'priority': RulePriority.MEDIUM,
                'pattern': r'\b(krishna|arjuna|rama|shiva)\b',
                'replacement': r'\1',
                'context_patterns': ['lord', 'god', 'deity', 'divine'],
                'conditions': {'check_capitalization': True},
                'confidence_boost': 0.05
            },
            {
                'id': 'scripture_context_consistency',
                'name': 'Scripture Context Consistency',
                'rule_type': RuleType.CATEGORY_CONSISTENCY,
                'priority': RulePriority.MEDIUM,
                'pattern': r'\b(gita|upanishad|veda|purana)\b',
                'replacement': r'\1',
                'context_patterns': ['text', 'scripture', 'verse', 'chapter'],
                'conditions': {'check_capitalization': True},
                'confidence_boost': 0.05
            }
        ]
        
        for rule_data in consistency_rules:
            rule = ContextualRule(**rule_data)
            self.rules[rule.id] = rule

    def _add_contextual_substitution_rules(self) -> None:
        """Add contextual substitution rules."""
        substitution_rules = [
            {
                'id': 'dharma_context_spelling',
                'name': 'Dharma Contextual Spelling',
                'rule_type': RuleType.CONTEXTUAL_SUBSTITUTION,
                'priority': RulePriority.MEDIUM,
                'pattern': r'\b(dh?arma?|dharama)\b',
                'replacement': 'dharma',
                'context_patterns': ['righteous', 'duty', 'virtue', 'law'],
                'confidence_boost': 0.1
            },
            {
                'id': 'yoga_context_spelling',
                'name': 'Yoga Contextual Spelling',
                'rule_type': RuleType.CONTEXTUAL_SUBSTITUTION,
                'priority': RulePriority.MEDIUM,
                'pattern': r'\b(yog[a]?|yogaa)\b',
                'replacement': 'yoga',
                'context_patterns': ['practice', 'path', 'union', 'discipline'],
                'confidence_boost': 0.1
            },
            {
                'id': 'meditation_context_spelling',
                'name': 'Meditation Contextual Spelling',
                'rule_type': RuleType.CONTEXTUAL_SUBSTITUTION,
                'priority': RulePriority.MEDIUM,
                'pattern': r'\b(dhyan[a]?|meditation)\b',
                'replacement': 'dhyana',
                'context_patterns': ['meditation', 'contemplation', 'practice'],
                'confidence_boost': 0.1
            }
        ]
        
        for rule_data in substitution_rules:
            rule = ContextualRule(**rule_data)
            self.rules[rule.id] = rule

    def _compile_rule_patterns(self) -> None:
        """Compile regex patterns for all rules."""
        for rule_id, rule in self.rules.items():
            try:
                self.compiled_patterns[rule_id] = re.compile(rule.pattern, re.IGNORECASE)
            except re.error as e:
                self.logger.error(f"Error compiling pattern for rule {rule_id}: {e}")

    def apply_contextual_rules(self, text: str, context_window: List[str] = None) -> List[RuleMatch]:
        """
        Apply all contextual rules to text.
        
        Args:
            text: Text to process
            context_window: Surrounding context words
            
        Returns:
            List of rule matches found and applied
        """
        matches = []
        context_window = context_window or []
        context_text = " ".join(context_window).lower()
        
        # Sort rules by priority
        sorted_rules = sorted(
            self.rules.items(), 
            key=lambda x: x[1].priority.value
        )
        
        processed_text = text
        
        for rule_id, rule in sorted_rules:
            if not rule.enabled:
                continue
                
            rule_matches = self._apply_single_rule(
                rule, processed_text, context_text, rule_id
            )
            
            for match in rule_matches:
                matches.append(match)
                # Apply the correction to continue processing with updated text
                processed_text = processed_text.replace(
                    match.original_text, match.corrected_text, 1
                )
        
        return matches

    def _apply_single_rule(self, rule: ContextualRule, text: str, 
                          context_text: str, rule_id: str) -> List[RuleMatch]:
        """Apply a single rule to text."""
        matches = []
        
        if rule_id not in self.compiled_patterns:
            return matches
        
        pattern = self.compiled_patterns[rule_id]
        
        # Check if context patterns match (if specified)
        context_matched = []
        if rule.context_patterns:
            for context_pattern in rule.context_patterns:
                if context_pattern.lower() in context_text:
                    context_matched.append(context_pattern)
            
            # If context patterns specified but none matched, skip rule
            if not context_matched:
                return matches
        
        # Find pattern matches
        for match in pattern.finditer(text):
            original_text = match.group(0)
            
            # Apply replacement
            if rule.rule_type == RuleType.COMPOUND_TERM:
                corrected_text = self._apply_compound_term_rule(rule, match)
            elif rule.rule_type == RuleType.SEQUENTIAL_DEPENDENCY:
                corrected_text = self._apply_sequential_dependency_rule(rule, match)
            elif rule.rule_type == RuleType.CATEGORY_CONSISTENCY:
                corrected_text = self._apply_category_consistency_rule(rule, match, context_text)
            else:
                corrected_text = pattern.sub(rule.replacement, original_text)
            
            # Calculate confidence
            base_confidence = 0.8
            confidence = base_confidence + rule.confidence_boost
            if context_matched:
                confidence += 0.1  # Bonus for context match
            
            confidence = min(1.0, confidence)
            
            rule_match = RuleMatch(
                rule_id=rule_id,
                rule_name=rule.name,
                original_text=original_text,
                corrected_text=corrected_text,
                match_position=match.start(),
                match_length=len(original_text),
                confidence=confidence,
                context_matched=context_matched,
                rule_type=rule.rule_type
            )
            
            matches.append(rule_match)
        
        return matches

    def _apply_compound_term_rule(self, rule: ContextualRule, match: re.Match) -> str:
        """Apply compound term standardization."""
        return rule.replacement

    def _apply_sequential_dependency_rule(self, rule: ContextualRule, match: re.Match) -> str:
        """Apply sequential dependency validation."""
        if 'validate_sequence' in rule.conditions and rule.conditions['validate_sequence']:
            # For chapter-verse sequences, validate the numbers are reasonable
            if 'chapter' in rule.pattern and 'verse' in rule.pattern:
                groups = match.groups()
                if len(groups) >= 4:
                    chapter_num = int(groups[1]) if groups[1].isdigit() else 0
                    verse_num = int(groups[3]) if groups[3].isdigit() else 0
                    
                    # Bhagavad Gita has 18 chapters, verses vary by chapter
                    if 1 <= chapter_num <= 18 and 1 <= verse_num <= 100:
                        return re.sub(rule.pattern, rule.replacement, match.group(0))
        
        return re.sub(rule.pattern, rule.replacement, match.group(0))

    def _apply_category_consistency_rule(self, rule: ContextualRule, match: re.Match, context: str) -> str:
        """Apply category consistency validation."""
        corrected = match.group(0)
        
        # Check capitalization for proper nouns
        if 'check_capitalization' in rule.conditions:
            # For deity names and scripture names, ensure proper capitalization
            if any(deity in match.group(0).lower() for deity in ['krishna', 'arjuna', 'rama', 'shiva']):
                corrected = corrected.title()
            elif any(scripture in match.group(0).lower() for scripture in ['gita', 'upanishad', 'veda']):
                corrected = corrected.title()
        
        return corrected

    def detect_compound_terms(self, text: str) -> List[CompoundTerm]:
        """
        Detect compound Sanskrit/Hindi terms in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected compound terms
        """
        compound_terms = []
        
        # Known compound term patterns
        compound_patterns = [
            (r'\b(karma)\s+(yoga?)\b', ['karma', 'yoga'], 'karma yoga'),
            (r'\b(bhakti)\s+(yoga?)\b', ['bhakti', 'yoga'], 'bhakti yoga'),
            (r'\b(raja?)\s+(yoga?)\b', ['raja', 'yoga'], 'raja yoga'),
            (r'\b(jnana?)\s+(yoga?)\b', ['jnana', 'yoga'], 'jnana yoga'),
            (r'\b(bhagavad?)\s+(gita?)\b', ['bhagavad', 'gita'], 'Bhagavad Gita'),
            (r'\b(sadhana)\s+(chatushtaya)\b', ['sadhana', 'chatushtaya'], 'sadhana chatushtaya')
        ]
        
        for pattern, components, canonical in compound_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                compound = CompoundTerm(
                    term=match.group(0),
                    components=components,
                    canonical_form=canonical,
                    transliteration=canonical,  # Would need IAST mapping
                    category='compound_term',
                    confidence=0.9
                )
                compound_terms.append(compound)
        
        return compound_terms

    def validate_contextual_dependencies(self, words: List[str], 
                                       positions: List[int]) -> List[Tuple[int, str, float]]:
        """
        Validate contextual dependencies between words.
        
        Args:
            words: List of words to validate
            positions: Positions of words in original text
            
        Returns:
            List of (position, issue_description, confidence) tuples
        """
        issues = []
        
        # Check for category consistency
        for i, word in enumerate(words):
            word_category = self._get_word_category(word)
            if word_category:
                # Check surrounding words for category consistency
                context_words = []
                start_idx = max(0, i - 2)
                end_idx = min(len(words), i + 3)
                
                for j in range(start_idx, end_idx):
                    if j != i:
                        context_words.append(words[j])
                
                # Look for category dependencies
                if word_category in self.category_dependencies:
                    expected_contexts = self.category_dependencies[word_category]
                    found_contexts = []
                    
                    for context_word in context_words:
                        if any(expected in context_word.lower() for expected in expected_contexts):
                            found_contexts.append(context_word)
                    
                    if not found_contexts:
                        issues.append((
                            positions[i],
                            f"Word '{word}' of category '{word_category}' lacks expected context",
                            0.6
                        ))
        
        return issues

    def _get_word_category(self, word: str) -> Optional[str]:
        """Get category for a word based on predefined mappings."""
        word_categories = {
            'krishna': 'deity',
            'arjuna': 'deity',
            'rama': 'deity',
            'shiva': 'deity',
            'gita': 'scripture',
            'upanishad': 'scripture',
            'veda': 'scripture',
            'yoga': 'practice',
            'meditation': 'practice',
            'dharma': 'concept',
            'karma': 'concept'
        }
        
        return word_categories.get(word.lower())

    def load_rules_from_config(self, config_file: Path) -> bool:
        """
        Load rules from configuration file.
        
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
            
            # Load rules from configuration
            if 'rules' in config_data:
                for rule_data in config_data['rules']:
                    rule = ContextualRule(
                        id=rule_data['id'],
                        name=rule_data['name'],
                        rule_type=RuleType(rule_data['rule_type']),
                        priority=RulePriority(rule_data.get('priority', 2)),
                        pattern=rule_data['pattern'],
                        replacement=rule_data['replacement'],
                        context_patterns=rule_data.get('context_patterns', []),
                        conditions=rule_data.get('conditions', {}),
                        confidence_boost=rule_data.get('confidence_boost', 0.0),
                        enabled=rule_data.get('enabled', True),
                        description=rule_data.get('description', ''),
                        examples=rule_data.get('examples', [])
                    )
                    self.rules[rule.id] = rule
            
            # Load category dependencies if present
            if 'category_dependencies' in config_data:
                self.category_dependencies.update(config_data['category_dependencies'])
            
            # Recompile patterns
            self._compile_rule_patterns()
            
            self.logger.info(f"Loaded {len(self.rules)} rules from {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading rules from {config_file}: {e}")
            return False

    def save_rules_to_config(self, config_file: Path) -> bool:
        """
        Save current rules to configuration file.
        
        Args:
            config_file: Path to save configuration
            
        Returns:
            True if successful
        """
        try:
            config_data = {
                'rules': [],
                'category_dependencies': self.category_dependencies
            }
            
            for rule in self.rules.values():
                rule_data = {
                    'id': rule.id,
                    'name': rule.name,
                    'rule_type': rule.rule_type.value,
                    'priority': rule.priority.value,
                    'pattern': rule.pattern,
                    'replacement': rule.replacement,
                    'context_patterns': rule.context_patterns,
                    'conditions': rule.conditions,
                    'confidence_boost': rule.confidence_boost,
                    'enabled': rule.enabled,
                    'description': rule.description,
                    'examples': rule.examples
                }
                config_data['rules'].append(rule_data)
            
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(self.rules)} rules to {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving rules to {config_file}: {e}")
            return False

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rule engine statistics."""
        rule_types = {}
        priority_distribution = {}
        enabled_count = 0
        
        for rule in self.rules.values():
            # Count by type
            rule_type = rule.rule_type.value
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
            
            # Count by priority
            priority = rule.priority.value
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
            
            # Count enabled
            if rule.enabled:
                enabled_count += 1
        
        return {
            'total_rules': len(self.rules),
            'enabled_rules': enabled_count,
            'disabled_rules': len(self.rules) - enabled_count,
            'rule_types': rule_types,
            'priority_distribution': priority_distribution,
            'category_dependencies': len(self.category_dependencies),
            'compiled_patterns': len(self.compiled_patterns)
        }