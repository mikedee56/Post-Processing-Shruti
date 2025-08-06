"""
N-gram Language Model for Contextual Prediction

This module provides statistical language modeling capabilities for Sanskrit/Hindi
context prediction using n-gram models built from corpus data.

Supports configurable n-gram orders (bigram, trigram, 4-gram) with smoothing
techniques for improved context likelihood scoring.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import math
import re
from enum import Enum
import logging

from utils.logger_config import get_logger


class SmoothingMethod(Enum):
    """Smoothing methods for n-gram models."""
    LAPLACE = "laplace"
    GOOD_TURING = "good_turing"
    KNESER_NEY = "kneser_ney"
    INTERPOLATION = "interpolation"


@dataclass
class NGramModelConfig:
    """Configuration for N-gram model."""
    n: int = 3  # N-gram order (bigram=2, trigram=3, etc.)
    smoothing_method: SmoothingMethod = SmoothingMethod.LAPLACE
    smoothing_parameter: float = 0.01
    min_count: int = 2
    vocabulary_size: int = 10000
    use_unk_token: bool = True
    unk_threshold: int = 1
    context_window_size: int = 10


@dataclass
class ContextPrediction:
    """Result of context-based prediction."""
    word: str
    probability: float
    log_probability: float
    context: List[str]
    confidence_score: float
    ngram_order: int


@dataclass
class NGramStatistics:
    """Statistics for n-gram model."""
    total_ngrams: int
    unique_ngrams: int
    vocabulary_size: int
    perplexity: float
    coverage: float
    model_size_mb: float


class NGramLanguageModel:
    """
    Statistical N-gram language model for contextual prediction.
    
    Builds and uses n-gram models to predict word likelihood in Sanskrit/Hindi
    context, supporting various smoothing techniques and configurable orders.
    """

    def __init__(self, config: NGramModelConfig = None):
        """
        Initialize N-gram language model.
        
        Args:
            config: Model configuration
        """
        self.config = config or NGramModelConfig()
        self.logger = get_logger(__name__)
        
        # Model data structures
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.ngram_probs: Dict[Tuple[str, ...], float] = {}
        self.vocabulary: Set[str] = set()
        self.word_counts: Dict[str, int] = defaultdict(int)
        
        # Special tokens
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"
        self.UNK_TOKEN = "<UNK>"
        
        # Model statistics
        self.statistics: Optional[NGramStatistics] = None
        self.is_trained = False
        
        self.logger.info(f"Initialized {self.config.n}-gram language model")

    def build_from_corpus(self, corpus_texts: List[str], 
                         domain_weight: float = 1.0) -> NGramStatistics:
        """
        Build n-gram model from corpus texts.
        
        Args:
            corpus_texts: List of text strings to build model from
            domain_weight: Weight for domain-specific texts
            
        Returns:
            NGramStatistics with model building results
        """
        self.logger.info(f"Building {self.config.n}-gram model from {len(corpus_texts)} texts")
        
        # Reset model state
        self.ngram_counts.clear()
        self.ngram_probs.clear()
        self.vocabulary.clear()
        self.word_counts.clear()
        
        # Process corpus
        total_tokens = 0
        for text in corpus_texts:
            tokens = self._tokenize_text(text)
            total_tokens += len(tokens)
            
            # Count words for vocabulary building
            for token in tokens:
                self.word_counts[token] += 1
            
            # Extract n-grams
            ngrams = self._extract_ngrams(tokens)
            for ngram in ngrams:
                self.ngram_counts[ngram] += int(domain_weight)
        
        # Build vocabulary with UNK handling
        self._build_vocabulary()
        
        # Calculate probabilities with smoothing
        self._calculate_probabilities()
        
        # Compute statistics
        self.statistics = self._compute_statistics(total_tokens)
        self.is_trained = True
        
        self.logger.info(f"Built model: {len(self.ngram_counts)} n-grams, "
                        f"vocab size: {len(self.vocabulary)}")
        
        return self.statistics

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words with preprocessing."""
        # Basic preprocessing
        text = text.lower().strip()
        
        # Handle punctuation and special characters
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Add start and end tokens for sentence boundaries
        if self.config.n > 1:
            start_tokens = [self.START_TOKEN] * (self.config.n - 1)
            end_tokens = [self.END_TOKEN] * (self.config.n - 1)
            tokens = start_tokens + tokens + end_tokens
        
        return tokens

    def _extract_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        """Extract n-grams from token sequence."""
        ngrams = []
        
        for i in range(len(tokens) - self.config.n + 1):
            ngram = tuple(tokens[i:i + self.config.n])
            ngrams.append(ngram)
        
        return ngrams

    def _build_vocabulary(self) -> None:
        """Build vocabulary with UNK token handling."""
        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), 
                            key=lambda x: x[1], reverse=True)
        
        # Add high-frequency words to vocabulary
        for word, count in sorted_words:
            if count >= self.config.unk_threshold:
                self.vocabulary.add(word)
            
            # Limit vocabulary size if specified
            if (self.config.vocabulary_size > 0 and 
                len(self.vocabulary) >= self.config.vocabulary_size):
                break
        
        # Add special tokens
        self.vocabulary.add(self.START_TOKEN)
        self.vocabulary.add(self.END_TOKEN)
        
        if self.config.use_unk_token:
            self.vocabulary.add(self.UNK_TOKEN)
        
        self.logger.info(f"Built vocabulary: {len(self.vocabulary)} words")

    def _map_to_vocabulary(self, tokens: List[str]) -> List[str]:
        """Map tokens to vocabulary, using UNK for out-of-vocabulary words."""
        if not self.config.use_unk_token:
            return tokens
        
        mapped_tokens = []
        for token in tokens:
            if token in self.vocabulary:
                mapped_tokens.append(token)
            else:
                mapped_tokens.append(self.UNK_TOKEN)
        
        return mapped_tokens

    def _calculate_probabilities(self) -> None:
        """Calculate n-gram probabilities with smoothing."""
        self.logger.info(f"Calculating probabilities using {self.config.smoothing_method.value}")
        
        if self.config.smoothing_method == SmoothingMethod.LAPLACE:
            self._apply_laplace_smoothing()
        elif self.config.smoothing_method == SmoothingMethod.INTERPOLATION:
            self._apply_interpolation_smoothing()
        else:
            # Default to Laplace for now
            self._apply_laplace_smoothing()

    def _apply_laplace_smoothing(self) -> None:
        """Apply Laplace (add-alpha) smoothing."""
        alpha = self.config.smoothing_parameter
        vocab_size = len(self.vocabulary)
        
        # Calculate (n-1)-gram counts for normalization
        context_counts = defaultdict(int)
        for ngram in self.ngram_counts:
            if len(ngram) > 1:
                context = ngram[:-1]
                context_counts[context] += self.ngram_counts[ngram]
        
        # Calculate smoothed probabilities
        for ngram in self.ngram_counts:
            if len(ngram) > 1:
                context = ngram[:-1]
                context_total = context_counts[context]
                
                # Laplace smoothing formula
                numerator = self.ngram_counts[ngram] + alpha
                denominator = context_total + (alpha * vocab_size)
                
                self.ngram_probs[ngram] = numerator / denominator
            else:
                # Unigram probability
                total_count = sum(self.ngram_counts.values())
                numerator = self.ngram_counts[ngram] + alpha
                denominator = total_count + (alpha * vocab_size)
                
                self.ngram_probs[ngram] = numerator / denominator

    def _apply_interpolation_smoothing(self) -> None:
        """Apply linear interpolation smoothing."""
        # For now, implement a simple version
        # In production, would use more sophisticated interpolation weights
        lambda_weights = [0.1, 0.3, 0.6]  # weights for 1-gram, 2-gram, 3-gram
        
        self._apply_laplace_smoothing()  # Base smoothing
        
        # Apply interpolation (simplified version)
        for ngram in list(self.ngram_probs.keys()):
            if len(ngram) == self.config.n:
                # Combine with lower-order probabilities
                interpolated_prob = self.ngram_probs[ngram]
                
                # Add lower-order probabilities with weights
                for i in range(1, len(ngram)):
                    lower_ngram = ngram[i:]
                    if lower_ngram in self.ngram_probs:
                        weight_idx = len(lower_ngram) - 1
                        if weight_idx < len(lambda_weights):
                            weight = lambda_weights[weight_idx]
                            interpolated_prob += weight * self.ngram_probs[lower_ngram]
                
                # Normalize
                self.ngram_probs[ngram] = min(interpolated_prob / sum(lambda_weights), 1.0)

    def predict_next_words(self, context: List[str], 
                          top_k: int = 10) -> List[ContextPrediction]:
        """
        Predict most likely next words given context.
        
        Args:
            context: Previous words for context
            top_k: Number of top predictions to return
            
        Returns:
            List of ContextPrediction objects ranked by probability
        """
        if not self.is_trained:
            self.logger.warning("Model not trained, returning empty predictions")
            return []
        
        # Prepare context
        context = self._map_to_vocabulary(context)
        
        # Get relevant context window
        if len(context) >= self.config.n - 1:
            context = context[-(self.config.n - 1):]
        else:
            # Pad with start tokens if needed
            padding = [self.START_TOKEN] * (self.config.n - 1 - len(context))
            context = padding + context
        
        # Find matching n-grams
        candidates = []
        context_tuple = tuple(context)
        
        for ngram in self.ngram_probs:
            if len(ngram) == self.config.n and ngram[:-1] == context_tuple:
                next_word = ngram[-1]
                probability = self.ngram_probs[ngram]
                log_prob = math.log(probability) if probability > 0 else float('-inf')
                
                # Calculate confidence score
                confidence = self._calculate_confidence(ngram, probability)
                
                prediction = ContextPrediction(
                    word=next_word,
                    probability=probability,
                    log_probability=log_prob,
                    context=list(context),
                    confidence_score=confidence,
                    ngram_order=self.config.n
                )
                candidates.append(prediction)
        
        # Sort by probability and return top-k
        candidates.sort(key=lambda x: x.probability, reverse=True)
        return candidates[:top_k]

    def calculate_sequence_probability(self, sequence: List[str]) -> float:
        """
        Calculate probability of a word sequence.
        
        Args:
            sequence: List of words
            
        Returns:
            Log probability of the sequence
        """
        if not self.is_trained:
            return float('-inf')
        
        sequence = self._map_to_vocabulary(sequence)
        ngrams = self._extract_ngrams(sequence)
        
        total_log_prob = 0.0
        
        for ngram in ngrams:
            if ngram in self.ngram_probs:
                prob = self.ngram_probs[ngram]
                if prob > 0:
                    total_log_prob += math.log(prob)
                else:
                    total_log_prob += float('-inf')
            else:
                # Use smoothing for unseen n-grams
                smoothed_prob = self.config.smoothing_parameter / len(self.vocabulary)
                total_log_prob += math.log(smoothed_prob)
        
        return total_log_prob

    def get_word_context_score(self, word: str, context: List[str]) -> float:
        """
        Get likelihood score for a word in given context.
        
        Args:
            word: Word to score
            context: Context words
            
        Returns:
            Context likelihood score (0-1)
        """
        if not self.is_trained:
            return 0.5  # Neutral score
        
        # Prepare context and word
        word = self._map_to_vocabulary([word])[0]
        context = self._map_to_vocabulary(context)
        
        # Try different context window sizes
        max_score = 0.0
        
        for window_size in range(1, min(len(context) + 1, self.config.n)):
            if len(context) >= window_size:
                test_context = context[-window_size:]
                test_ngram = tuple(test_context + [word])
                
                if test_ngram in self.ngram_probs:
                    prob = self.ngram_probs[test_ngram]
                    max_score = max(max_score, prob)
        
        return max_score

    def _calculate_confidence(self, ngram: Tuple[str, ...], probability: float) -> float:
        """Calculate confidence score for a prediction."""
        # Base confidence from probability
        base_confidence = probability
        
        # Adjust based on n-gram frequency
        count = self.ngram_counts.get(ngram, 0)
        frequency_boost = min(count / 10.0, 0.3)  # Max 0.3 boost
        
        # Adjust based on vocabulary presence
        vocab_penalty = 0.0
        for word in ngram:
            if word == self.UNK_TOKEN:
                vocab_penalty += 0.2
        
        confidence = base_confidence + frequency_boost - vocab_penalty
        return max(0.0, min(1.0, confidence))

    def _compute_statistics(self, total_tokens: int) -> NGramStatistics:
        """Compute model statistics."""
        unique_ngrams = len(self.ngram_counts)
        total_ngrams = sum(self.ngram_counts.values())
        vocab_size = len(self.vocabulary)
        
        # Calculate perplexity (simplified)
        avg_log_prob = sum(math.log(p) for p in self.ngram_probs.values()) / len(self.ngram_probs)
        perplexity = math.exp(-avg_log_prob) if avg_log_prob != 0 else float('inf')
        
        # Calculate coverage
        coverage = vocab_size / max(len(self.word_counts), 1)
        
        # Estimate model size
        model_size_mb = (len(self.ngram_counts) * 100 + len(self.ngram_probs) * 100) / (1024 * 1024)
        
        return NGramStatistics(
            total_ngrams=total_ngrams,
            unique_ngrams=unique_ngrams,
            vocabulary_size=vocab_size,
            perplexity=perplexity,
            coverage=coverage,
            model_size_mb=model_size_mb
        )

    def save_model(self, file_path: Path) -> bool:
        """
        Save model to file.
        
        Args:
            file_path: Path to save model
            
        Returns:
            True if successful
        """
        try:
            model_data = {
                'config': asdict(self.config),
                'ngram_counts': dict(self.ngram_counts),
                'ngram_probs': dict(self.ngram_probs),
                'vocabulary': list(self.vocabulary),
                'word_counts': dict(self.word_counts),
                'statistics': asdict(self.statistics) if self.statistics else None,
                'is_trained': self.is_trained
            }
            
            # Save as pickle for efficiency
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Saved model to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, file_path: Path) -> bool:
        """
        Load model from file.
        
        Args:
            file_path: Path to model file
            
        Returns:
            True if successful
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model state
            self.config = NGramModelConfig(**model_data['config'])
            self.ngram_counts = defaultdict(int, model_data['ngram_counts'])
            self.ngram_probs = model_data['ngram_probs']
            self.vocabulary = set(model_data['vocabulary'])
            self.word_counts = defaultdict(int, model_data['word_counts'])
            self.is_trained = model_data['is_trained']
            
            if model_data['statistics']:
                self.statistics = NGramStatistics(**model_data['statistics'])
            
            self.logger.info(f"Loaded model from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'config': asdict(self.config),
            'statistics': asdict(self.statistics) if self.statistics else None,
            'is_trained': self.is_trained,
            'vocabulary_size': len(self.vocabulary),
            'ngram_count': len(self.ngram_counts),
            'model_type': f'{self.config.n}-gram',
            'smoothing_method': self.config.smoothing_method.value
        }