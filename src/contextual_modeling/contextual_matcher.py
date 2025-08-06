"""
Contextual Matching System

This module integrates n-gram language models with the existing fuzzy matching
system to provide context-aware correction suggestions for Sanskrit/Hindi terms.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from utils.fuzzy_matcher import FuzzyMatcher, FuzzyMatch, MatchingConfig, MatchType
from sanskrit_hindi_identifier.lexicon_manager import LexiconManager
from contextual_modeling.ngram_language_model import NGramLanguageModel, NGramModelConfig, ContextPrediction
from utils.logger_config import get_logger


@dataclass
class ContextualMatchConfig:
    """Configuration for contextual matching."""
    fuzzy_config: MatchingConfig
    ngram_config: NGramModelConfig
    context_weight: float = 0.3
    fuzzy_weight: float = 0.7
    min_combined_confidence: float = 0.6
    max_context_window: int = 10
    enable_context_filtering: bool = True
    context_boost_threshold: float = 0.8


@dataclass
class ContextualMatch:
    """Enhanced fuzzy match with contextual information."""
    fuzzy_match: FuzzyMatch
    context_score: float
    combined_confidence: float
    context_prediction: Optional[ContextPrediction] = None
    context_boost: float = 0.0


class ContextualMatcher:
    """
    Context-aware fuzzy matcher that combines traditional fuzzy matching
    with n-gram language model predictions for improved accuracy.
    """

    def __init__(self, lexicon_manager: LexiconManager, 
                 config: ContextualMatchConfig = None):
        """
        Initialize contextual matcher.
        
        Args:
            lexicon_manager: Lexicon management system
            config: Contextual matching configuration
        """
        self.logger = get_logger(__name__)
        self.lexicon_manager = lexicon_manager
        self.config = config or self._get_default_config()
        
        # Initialize fuzzy matcher with lexicon data
        all_entries = self.lexicon_manager.get_all_entries()
        lexicon_data = {}
        for term, entry in all_entries.items():
            lexicon_data[term] = {
                'transliteration': entry.transliteration,
                'is_proper_noun': entry.is_proper_noun,
                'category': entry.category,
                'confidence': entry.confidence,
                'source_authority': entry.source_authority,
                'variations': entry.variations
            }
        
        self.fuzzy_matcher = FuzzyMatcher(lexicon_data, self.config.fuzzy_config)
        
        # Initialize n-gram language model
        self.ngram_model = NGramLanguageModel(self.config.ngram_config)
        
        # Model training status
        self.is_model_trained = False
        
        self.logger.info("ContextualMatcher initialized")

    def _get_default_config(self) -> ContextualMatchConfig:
        """Get default configuration."""
        return ContextualMatchConfig(
            fuzzy_config=MatchingConfig(),
            ngram_config=NGramModelConfig()
        )

    def train_context_model(self, corpus_texts: List[str]) -> bool:
        """
        Train the n-gram context model on corpus texts.
        
        Args:
            corpus_texts: List of training texts
            
        Returns:
            True if training successful
        """
        try:
            self.logger.info(f"Training context model with {len(corpus_texts)} texts")
            
            # Add Sanskrit/Hindi specific texts by extracting terms from lexicon
            enhanced_corpus = corpus_texts.copy()
            
            # Generate synthetic training data from lexicon
            lexicon_sentences = self._generate_lexicon_sentences()
            enhanced_corpus.extend(lexicon_sentences)
            
            # Train the model
            statistics = self.ngram_model.build_from_corpus(enhanced_corpus, domain_weight=1.5)
            
            self.is_model_trained = True
            self.logger.info(f"Context model trained: {statistics.unique_ngrams} n-grams, "
                           f"perplexity: {statistics.perplexity:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training context model: {e}")
            return False

    def _generate_lexicon_sentences(self) -> List[str]:
        """Generate training sentences from lexicon entries."""
        sentences = []
        all_entries = self.lexicon_manager.get_all_entries()
        
        # Group terms by category for contextual sentences
        categories = {}
        for term, entry in all_entries.items():
            category = entry.category
            if category not in categories:
                categories[category] = []
            categories[category].append(entry.original_term)
        
        # Generate synthetic sentences for each category
        category_templates = {
            'scripture': [
                "The {term} teaches us about dharma and righteousness.",
                "We study the {term} to understand spiritual wisdom.",
                "In the {term}, we find guidance for life."
            ],
            'deity': [
                "Lord {term} is revered in Hindu tradition.",
                "{term} is worshipped with devotion.",
                "The teachings of {term} guide devotees."
            ],
            'concept': [
                "The concept of {term} is fundamental to yoga philosophy.",
                "Understanding {term} leads to spiritual growth.",
                "{term} is explained in ancient texts."
            ],
            'practice': [
                "The practice of {term} brings inner peace.",
                "Students learn {term} meditation techniques.",
                "{term} is taught in traditional ashrams."
            ]
        }
        
        # Generate sentences for each category
        for category, terms in categories.items():
            templates = category_templates.get(category, [
                "The {term} is important in spiritual practice.",
                "We learn about {term} in our studies.",
                "{term} is a key concept."
            ])
            
            for term in terms[:10]:  # Limit to avoid too much synthetic data
                for template in templates:
                    sentence = template.format(term=term)
                    sentences.append(sentence)
        
        return sentences

    def find_contextual_matches(self, word: str, context: List[str], 
                              max_matches: int = 5) -> List[ContextualMatch]:
        """
        Find contextual matches combining fuzzy matching with n-gram predictions.
        
        Args:
            word: Word to match
            context: Context words (previous words in sequence)
            max_matches: Maximum matches to return
            
        Returns:
            List of contextual matches ranked by combined confidence
        """
        # Get fuzzy matches first
        context_str = " ".join(context[-self.config.max_context_window:])
        fuzzy_matches = self.fuzzy_matcher.find_matches(word, context_str, max_matches * 2)
        
        contextual_matches = []
        
        for fuzzy_match in fuzzy_matches:
            # Calculate context score if model is trained
            context_score = 0.5  # Default neutral score
            context_prediction = None
            context_boost = 0.0
            
            if self.is_model_trained and len(context) > 0:
                # Get context score for the corrected term
                context_score = self.ngram_model.get_word_context_score(
                    fuzzy_match.corrected_term, context
                )
                
                # Get context predictions to see if this word is likely
                predictions = self.ngram_model.predict_next_words(context, top_k=10)
                for pred in predictions:
                    if pred.word.lower() == fuzzy_match.corrected_term.lower():
                        context_prediction = pred
                        # Boost confidence if word appears in top predictions
                        if pred.probability > self.config.context_boost_threshold:
                            context_boost = min(0.2, pred.probability * 0.3)
                        break
            
            # Calculate combined confidence
            fuzzy_confidence = fuzzy_match.confidence
            combined_confidence = (
                self.config.fuzzy_weight * fuzzy_confidence + 
                self.config.context_weight * context_score +
                context_boost
            )
            
            # Apply context filtering if enabled
            if self.config.enable_context_filtering:
                if context_score < 0.1 and fuzzy_confidence < 0.9:
                    # Very low context score - reduce confidence
                    combined_confidence *= 0.7
                elif context_score > 0.7:
                    # High context score - boost confidence
                    combined_confidence = min(1.0, combined_confidence * 1.1)
            
            contextual_match = ContextualMatch(
                fuzzy_match=fuzzy_match,
                context_score=context_score,
                combined_confidence=combined_confidence,
                context_prediction=context_prediction,
                context_boost=context_boost
            )
            
            contextual_matches.append(contextual_match)
        
        # Sort by combined confidence and filter
        contextual_matches.sort(key=lambda m: m.combined_confidence, reverse=True)
        
        # Filter by minimum confidence
        filtered_matches = [
            m for m in contextual_matches 
            if m.combined_confidence >= self.config.min_combined_confidence
        ]
        
        return filtered_matches[:max_matches]

    def predict_likely_corrections(self, context: List[str], 
                                 top_k: int = 5) -> List[ContextPrediction]:
        """
        Predict likely next words based on context.
        
        Args:
            context: Context words
            top_k: Number of predictions
            
        Returns:
            List of context predictions
        """
        if not self.is_model_trained:
            return []
        
        predictions = self.ngram_model.predict_next_words(context, top_k)
        
        # Filter predictions to include only lexicon terms
        lexicon_predictions = []
        all_entries = self.lexicon_manager.get_all_entries()
        
        for pred in predictions:
            if pred.word.lower() in all_entries:
                lexicon_predictions.append(pred)
        
        return lexicon_predictions

    def calculate_sequence_likelihood(self, words: List[str]) -> float:
        """
        Calculate likelihood of a word sequence.
        
        Args:
            words: Sequence of words
            
        Returns:
            Log likelihood score
        """
        if not self.is_model_trained:
            return 0.0
        
        return self.ngram_model.calculate_sequence_probability(words)

    def get_context_suggestions(self, partial_text: str, 
                              context: List[str]) -> List[str]:
        """
        Get contextual suggestions for partial text.
        
        Args:
            partial_text: Incomplete text
            context: Previous context
            
        Returns:
            List of suggested completions
        """
        suggestions = []
        
        # Get fuzzy matches for partial text
        fuzzy_matches = self.fuzzy_matcher.find_matches(partial_text, " ".join(context))
        
        # Get context predictions
        if self.is_model_trained:
            context_predictions = self.ngram_model.predict_next_words(context, top_k=10)
            
            # Combine fuzzy matches with context predictions
            combined_suggestions = set()
            
            for match in fuzzy_matches[:5]:
                combined_suggestions.add(match.corrected_term)
            
            for pred in context_predictions:
                combined_suggestions.add(pred.word)
            
            suggestions = list(combined_suggestions)
        else:
            suggestions = [m.corrected_term for m in fuzzy_matches[:5]]
        
        return suggestions

    def save_context_model(self, file_path: Path) -> bool:
        """
        Save trained context model.
        
        Args:
            file_path: Path to save model
            
        Returns:
            True if successful
        """
        if not self.is_model_trained:
            self.logger.warning("No trained model to save")
            return False
        
        return self.ngram_model.save_model(file_path)

    def load_context_model(self, file_path: Path) -> bool:
        """
        Load trained context model.
        
        Args:
            file_path: Path to model file
            
        Returns:
            True if successful
        """
        success = self.ngram_model.load_model(file_path)
        if success:
            self.is_model_trained = True
        
        return success

    def get_matching_statistics(self) -> Dict[str, Any]:
        """Get comprehensive matching statistics."""
        stats = {
            'fuzzy_matcher': {
                'terms_indexed': len(self.fuzzy_matcher.search_terms),
                'config': {
                    'min_confidence': self.config.fuzzy_config.min_confidence,
                    'levenshtein_threshold': self.config.fuzzy_config.levenshtein_threshold,
                    'enable_phonetic': self.config.fuzzy_config.enable_phonetic_matching
                }
            },
            'context_model': {
                'is_trained': self.is_model_trained,
                'model_info': self.ngram_model.get_model_info() if self.is_model_trained else None
            },
            'contextual_config': {
                'context_weight': self.config.context_weight,
                'fuzzy_weight': self.config.fuzzy_weight,
                'min_combined_confidence': self.config.min_combined_confidence,
                'enable_context_filtering': self.config.enable_context_filtering
            }
        }
        
        return stats