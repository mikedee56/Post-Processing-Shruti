"""
Scripture Processing Module.

This module provides comprehensive capabilities for identifying and substituting
scriptural verses with canonical text in ASR transcripts.
"""

from scripture_processing.scripture_identifier import ScriptureIdentifier
from scripture_processing.canonical_text_manager import CanonicalTextManager
from scripture_processing.verse_substitution_engine import VerseSubstitutionEngine
from scripture_processing.scripture_validator import ScriptureValidator

__all__ = [
    "ScriptureIdentifier",
    "CanonicalTextManager", 
    "VerseSubstitutionEngine",
    "ScriptureValidator"
]