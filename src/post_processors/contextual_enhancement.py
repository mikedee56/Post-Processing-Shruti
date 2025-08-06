"""
Contextual Enhancement Integration for SanskritPostProcessor

This module provides integration of Story 2.2 contextual modeling components
with the existing SanskritPostProcessor to enable context-aware corrections.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from contextual_modeling.ngram_language_model import NGramLanguageModel, NGramModelConfig
from contextual_modeling.phonetic_encoder import PhoneticEncoder, PhoneticConfig
from contextual_modeling.contextual_rule_engine import ContextualRuleEngine
from contextual_modeling.spelling_normalizer import SpellingNormalizer
from contextual_modeling.contextual_matcher import ContextualMatcher, ContextualMatchConfig
from sanskrit_hindi_identifier.lexicon_manager import LexiconManager
from utils.logger_config import get_logger
from utils.srt_parser import SRTSegment


@dataclass
class ContextualProcessingResult:
    """Result of contextual processing enhancement."""
    original_text: str
    enhanced_text: str
    confidence_score: float
    contextual_changes: List[Tuple[str, str, str]]  # (original, replacement, reason)
    ngram_predictions: List[str]
    phonetic_matches: List[str]
    rule_applications: List[str]
    spelling_normalizations: List[str]
    processing_time: float


class ContextualEnhancement:
    """
    Contextual enhancement system that integrates all Story 2.2 components.
    
    Provides context-aware corrections using n-gram models, phonetic encoding,
    rule engines, and spelling normalization.
    """

    def __init__(self, lexicon_manager: LexiconManager, 
                 contextual_config_path: Optional[Path] = None):
        """
        Initialize contextual enhancement system.
        
        Args:
            lexicon_manager: Existing lexicon management system
            contextual_config_path: Path to contextual configuration file
        """
        self.logger = get_logger(__name__)
        self.lexicon_manager = lexicon_manager
        
        # Load contextual configuration
        config_path = contextual_config_path or Path("config/contextual_config.yaml")
        self.config = self._load_contextual_config(config_path)
        
        # Initialize contextual components
        self._initialize_contextual_components()
        
        # Training status
        self.is_context_model_trained = False
        
        self.logger.info("ContextualEnhancement initialized with all Story 2.2 components")

    def _load_contextual_config(self, config_path: Path) -> Dict[str, Any]:
        """Load contextual modeling configuration."""
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.suffix.lower() in ['.yaml', '.yml']:
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                
                self.logger.info(f"Loaded contextual config from {config_path}")
                return config
            else:
                self.logger.warning(f"Config file {config_path} not found, using defaults")
                return self._get_default_contextual_config()
        except Exception as e:
            self.logger.error(f"Error loading contextual config: {e}")
            return self._get_default_contextual_config()

    def _get_default_contextual_config(self) -> Dict[str, Any]:
        """Get default contextual configuration."""
        return {
            'ngram_model': {'n': 3, 'smoothing_method': 'laplace'},
            'phonetic_encoder': {'algorithm': 'sanskrit_phonetic'},
            'contextual_matcher': {'context_weight': 0.3, 'fuzzy_weight': 0.7},
            'processing_pipeline': {'enable_ngram_context': True}
        }

    def _initialize_contextual_components(self) -> None:
        """Initialize all contextual modeling components."""
        # N-gram Language Model
        ngram_config = NGramModelConfig(**self.config.get('ngram_model', {}))
        self.ngram_model = NGramLanguageModel(ngram_config)
        
        # Phonetic Encoder
        phonetic_config = PhoneticConfig(**self.config.get('phonetic_encoder', {}))
        self.phonetic_encoder = PhoneticEncoder(phonetic_config)
        
        # Contextual Rule Engine
        rules_config_path = None
        if 'contextual_rules' in self.config:
            # Save rules to temporary file for engine to load
            rules_config_path = Path("config/temp_contextual_rules.yaml")
            with open(rules_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config['contextual_rules'], f, default_flow_style=False)
        
        self.rule_engine = ContextualRuleEngine(rules_config_path)
        
        # Spelling Normalizer
        self.spelling_normalizer = SpellingNormalizer()
        if 'spelling_normalization' in self.config:
            # Configure spelling normalizer with config data
            self.spelling_normalizer.normalization_rules.clear()
            self.spelling_normalizer.expansion_mappings.update(
                self.config['spelling_normalization'].get('expansion_mappings', {})
            )
            self.spelling_normalizer.variant_groups.update(
                self.config['spelling_normalization'].get('variant_groups', {})
            )
        
        # Contextual Matcher (integrates fuzzy matching with n-gram context)
        contextual_config = ContextualMatchConfig(
            **self.config.get('contextual_matcher', {})
        )
        self.contextual_matcher = ContextualMatcher(
            self.lexicon_manager, contextual_config
        )

    def train_contextual_models(self, training_texts: List[str]) -> bool:
        """
        Train contextual models with corpus data.
        
        Args:
            training_texts: List of training text strings
            
        Returns:
            True if training successful
        """
        try:
            self.logger.info(f"Training contextual models with {len(training_texts)} texts")
            
            # Train n-gram model directly
            self.ngram_model.build_from_corpus(training_texts)
            
            # Train contextual matcher (which uses the n-gram model)
            success = self.contextual_matcher.train_context_model(training_texts)
            
            if success:
                self.is_context_model_trained = True
                self.logger.info("Contextual models training completed successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error training contextual models: {e}")
            return False

    def apply_contextual_enhancement(self, segment: SRTSegment, 
                                   context_segments: List[SRTSegment] = None) -> ContextualProcessingResult:
        """
        Apply contextual enhancement to an SRT segment.
        
        Args:
            segment: SRT segment to enhance
            context_segments: Previous segments for context
            
        Returns:
            ContextualProcessingResult with enhancement details
        """
        import time
        start_time = time.time()
        
        original_text = segment.text
        enhanced_text = original_text
        contextual_changes = []
        ngram_predictions = []
        phonetic_matches = []
        rule_applications = []
        spelling_normalizations = []
        
        # Extract context from previous segments
        context_words = []
        if context_segments:
            for ctx_segment in context_segments[-3:]:  # Use last 3 segments
                context_words.extend(ctx_segment.text.lower().split())
            context_words = context_words[-20:]  # Limit context window
        
        # Processing pipeline based on configuration order
        pipeline_config = self.config.get('processing_pipeline', {})
        
        # Step 1: Apply contextual rules
        if pipeline_config.get('enable_contextual_rules', True):
            rule_matches = self.rule_engine.apply_contextual_rules(
                enhanced_text, context_words
            )
            for match in rule_matches:
                enhanced_text = enhanced_text.replace(
                    match.original_text, match.corrected_text, 1
                )
                contextual_changes.append((
                    match.original_text, 
                    match.corrected_text, 
                    f"Rule: {match.rule_name}"
                ))
                rule_applications.append(match.rule_name)
        
        # Step 2: Apply phonetic matching
        if pipeline_config.get('enable_phonetic_matching', True) and self.is_context_model_trained:
            words = enhanced_text.split()
            for i, word in enumerate(words):
                # Get contextual matches that include phonetic scoring
                contextual_matches = self.contextual_matcher.find_contextual_matches(
                    word, context_words, max_matches=3
                )
                
                if contextual_matches and contextual_matches[0].combined_confidence > 0.7:
                    best_match = contextual_matches[0]
                    if best_match.fuzzy_match.match_type.value == 'phonetic':
                        original_word = word
                        corrected_word = best_match.fuzzy_match.corrected_term
                        enhanced_text = enhanced_text.replace(original_word, corrected_word, 1)
                        contextual_changes.append((
                            original_word,
                            corrected_word,
                            f"Phonetic: {best_match.fuzzy_match.confidence:.2f}"
                        ))
                        phonetic_matches.append(corrected_word)
        
        # Step 3: Apply n-gram context predictions
        if pipeline_config.get('enable_ngram_context', True) and self.is_context_model_trained:
            # Get predictions for potential improvements
            predictions = self.ngram_model.predict_next_words(context_words, top_k=5)
            ngram_predictions = [pred.word for pred in predictions]
            
            # Use predictions to validate/boost corrections
            words = enhanced_text.split()
            for i, word in enumerate(words):
                word_score = self.ngram_model.get_word_context_score(word, context_words)
                
                # If word has low context score, check if predictions offer better alternatives
                if word_score < 0.3 and predictions:
                    for pred in predictions[:3]:  # Check top 3 predictions
                        if self._is_similar_word(word, pred.word) and pred.confidence_score > 0.8:
                            enhanced_text = enhanced_text.replace(word, pred.word, 1)
                            contextual_changes.append((
                                word,
                                pred.word,
                                f"N-gram context: {pred.confidence_score:.2f}"
                            ))
                            break
        
        # Step 4: Apply spelling normalization
        if pipeline_config.get('enable_spelling_normalization', True):
            normalization_result = self.spelling_normalizer.normalize_text(
                enhanced_text, context_words, document_id=f"segment_{segment.index}"
            )
            
            if normalization_result.changes_made:
                enhanced_text = normalization_result.normalized_text
                for original, replacement, pos in normalization_result.changes_made:
                    contextual_changes.append((
                        original,
                        replacement,
                        "Spelling normalization"
                    ))
                    spelling_normalizations.append(f"{original} -> {replacement}")
        
        # Calculate overall confidence
        confidence_score = self._calculate_enhancement_confidence(
            original_text, enhanced_text, contextual_changes, context_words
        )
        
        processing_time = time.time() - start_time
        
        return ContextualProcessingResult(
            original_text=original_text,
            enhanced_text=enhanced_text,
            confidence_score=confidence_score,
            contextual_changes=contextual_changes,
            ngram_predictions=ngram_predictions,
            phonetic_matches=phonetic_matches,
            rule_applications=rule_applications,
            spelling_normalizations=spelling_normalizations,
            processing_time=processing_time
        )

    def _is_similar_word(self, word1: str, word2: str) -> bool:
        """Check if two words are similar enough to consider substitution."""
        # Simple similarity check - could be enhanced with more sophisticated metrics
        if len(word1) != len(word2):
            return False
        
        differences = sum(c1 != c2 for c1, c2 in zip(word1.lower(), word2.lower()))
        return differences <= 2  # Allow up to 2 character differences

    def _calculate_enhancement_confidence(self, original: str, enhanced: str, 
                                        changes: List[Tuple[str, str, str]], 
                                        context: List[str]) -> float:
        """Calculate confidence score for contextual enhancement."""
        base_confidence = 0.7
        
        # Boost for context presence
        if context and len(context) > 5:
            base_confidence += 0.1
        
        # Adjust based on number and type of changes
        if changes:
            rule_changes = sum(1 for _, _, reason in changes if 'Rule:' in reason)
            phonetic_changes = sum(1 for _, _, reason in changes if 'Phonetic:' in reason)
            ngram_changes = sum(1 for _, _, reason in changes if 'N-gram' in reason)
            
            # Boost for high-confidence change types
            base_confidence += rule_changes * 0.05
            base_confidence += phonetic_changes * 0.08
            base_confidence += ngram_changes * 0.1
            
            # Slight penalty for too many changes
            if len(changes) > 5:
                base_confidence -= 0.05
        else:
            # If no changes made, moderate confidence
            base_confidence = 0.6
        
        # Context model training boost
        if self.is_context_model_trained:
            base_confidence += 0.05
        
        return max(0.0, min(1.0, base_confidence))

    def get_contextual_suggestions(self, text: str, context: List[str] = None) -> List[str]:
        """
        Get contextual suggestions for text improvement.
        
        Args:
            text: Text to get suggestions for
            context: Context words
            
        Returns:
            List of suggested improvements
        """
        suggestions = []
        context = context or []
        
        if self.is_context_model_trained:
            # Get n-gram predictions
            predictions = self.ngram_model.predict_next_words(context, top_k=5)
            suggestions.extend([pred.word for pred in predictions])
            
            # Get contextual matcher suggestions
            words = text.split()
            for word in words[-3:]:  # Check last few words
                contextual_suggestions = self.contextual_matcher.get_context_suggestions(
                    word, context
                )
                suggestions.extend(contextual_suggestions)
        
        # Get spelling suggestions
        words = text.split()
        for word in words:
            spelling_suggestions = self.spelling_normalizer.get_spelling_suggestions(word, 3)
            suggestions.extend([sugg[0] for sugg in spelling_suggestions])
        
        # Remove duplicates and return
        return list(set(suggestions))

    def save_contextual_models(self, models_dir: Path) -> bool:
        """
        Save trained contextual models to directory.
        
        Args:
            models_dir: Directory to save models
            
        Returns:
            True if successful
        """
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save n-gram model
            ngram_path = models_dir / "ngram_model.pkl"
            self.ngram_model.save_model(ngram_path)
            
            # Save contextual matcher model
            contextual_path = models_dir / "contextual_model.pkl"
            self.contextual_matcher.save_context_model(contextual_path)
            
            # Save phonetic mappings
            phonetic_path = models_dir / "phonetic_mappings.json"
            lexicon_entries = self.lexicon_manager.get_all_entries()
            phonetic_codes = self.phonetic_encoder.encode_lexicon_batch({
                term: {'variations': entry.variations}
                for term, entry in lexicon_entries.items()
            })
            self.phonetic_encoder.save_phonetic_mappings(phonetic_path, phonetic_codes)
            
            self.logger.info(f"Saved contextual models to {models_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving contextual models: {e}")
            return False

    def load_contextual_models(self, models_dir: Path) -> bool:
        """
        Load trained contextual models from directory.
        
        Args:
            models_dir: Directory containing models
            
        Returns:
            True if successful
        """
        try:
            # Load n-gram model
            ngram_path = models_dir / "ngram_model.pkl"
            if ngram_path.exists():
                success = self.ngram_model.load_model(ngram_path)
                if success:
                    self.is_context_model_trained = True
            
            # Load contextual matcher model
            contextual_path = models_dir / "contextual_model.pkl"
            if contextual_path.exists():
                self.contextual_matcher.load_context_model(contextual_path)
            
            # Load phonetic mappings
            phonetic_path = models_dir / "phonetic_mappings.json"
            if phonetic_path.exists():
                self.phonetic_encoder.load_phonetic_mappings(phonetic_path)
            
            self.logger.info(f"Loaded contextual models from {models_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading contextual models: {e}")
            return False

    def get_contextual_statistics(self) -> Dict[str, Any]:
        """Get comprehensive contextual enhancement statistics."""
        return {
            'ngram_model': self.ngram_model.get_model_info(),
            'phonetic_encoder': {
                'algorithm': self.phonetic_encoder.config.algorithm.value,
                'max_code_length': self.phonetic_encoder.config.max_code_length
            },
            'rule_engine': self.rule_engine.get_rule_statistics(),
            'spelling_normalizer': self.spelling_normalizer.get_normalization_statistics(),
            'contextual_matcher': self.contextual_matcher.get_matching_statistics(),
            'is_context_model_trained': self.is_context_model_trained
        }