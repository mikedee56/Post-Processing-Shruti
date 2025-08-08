"""
Enhancement Integration Package

This package provides cross-story enhancement coordination for the advanced 
ASR post-processing workflow, implementing Story 2.4.4 requirements.

Components:
- UnifiedConfidenceScorer: System-wide confidence scoring normalization
- ProvenanceManager: Gold/Silver/Bronze source classification and weighting
- CrossStoryCoordinator: Integration coordination layer
- FeatureFlags: Enhancement enable/disable control
"""

__version__ = "2.4.4"