"""
Scripture Processing Module.

This module provides comprehensive capabilities for identifying and substituting
scriptural verses with canonical text in ASR transcripts.
"""

from .scripture_identifier import ScriptureIdentifier
from .canonical_text_manager import CanonicalTextManager
from .verse_substitution_engine import VerseSubstitutionEngine
from .scripture_validator import ScriptureValidator

__all__ = [
    "ScriptureIdentifier",
    "CanonicalTextManager", 
    "VerseSubstitutionEngine",
    "ScriptureValidator"
]