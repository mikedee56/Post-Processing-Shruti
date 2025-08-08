"""
Research Integration Package

This package provides research validation, performance benchmarking, and lexicon acquisition
capabilities for the Post-Processing-Shruti project. It supports continuous quality assurance
and research-grade validation of enhanced processing algorithms.

Components:
- PerformanceBenchmarking: Comprehensive performance analysis framework
- ResearchValidationMetrics: Academic accuracy validation tools
- LexiconAcquisition: Multi-source lexicon building and quality assessment
- BenchmarkingSuite: Automated validation and regression testing
"""

from .performance_benchmarking import PerformanceBenchmarking
from .research_validation_metrics import ResearchValidationMetrics
from .lexicon_acquisition import LexiconAcquisition

__all__ = [
    'PerformanceBenchmarking',
    'ResearchValidationMetrics', 
    'LexiconAcquisition'
]