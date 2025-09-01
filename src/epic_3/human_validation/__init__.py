"""
Human Validation Module - Story 3.3 & 3.3.1

This module implements the expert review dashboard and knowledge capture system
for continuous learning from human expert decisions.

Components:
- knowledge_capture: Expert decision storage and pattern extraction
- validation_models: Data models for expert validation cases
"""

from .knowledge_capture import KnowledgeCapture
from .validation_models import ValidationCase, ExpertDecision, LearningPattern

__all__ = [
    'KnowledgeCapture',
    'ValidationCase', 
    'ExpertDecision',
    'LearningPattern',
]