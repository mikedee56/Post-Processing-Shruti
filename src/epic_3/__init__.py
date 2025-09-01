"""
Epic 3: Semantic Refinement & QA Framework

This module contains advanced semantic processing and human validation components
for achieving academic excellence in Sanskrit/Hindi post-processing.

Components:
- human_validation: Expert review and knowledge capture systems
- semantic_processing: Advanced semantic analysis and relationship modeling
- academic_validation: Quality gates and academic standards compliance
"""

__version__ = "1.0.0"
__epic__ = "Epic 3: Semantic Refinement & QA Framework"

# Module imports for Epic 3 components
from .human_validation import *
from .academic_validation import *

__all__ = [
    'human_validation',
    'academic_validation',
]