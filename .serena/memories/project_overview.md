# Project Overview

## Purpose
Advanced ASR Post-Processing Workflow for transforming ASR-generated transcripts of Yoga Vedanta lectures into highly accurate, academically rigorous textual resources. The system processes SRT files to correct Sanskrit/Hindi terms, apply IAST transliteration standards, and identify scriptural verses.

## Tech Stack
- **Language**: Python 3.10+ (system has 3.12.3)
- **Data Processing**: pandas, numpy, scipy
- **NLP Libraries**: iNLTK, IndicNLP Library for Indic language support
- **Specialized**: ByT5-Sanskrit (optional), sanskrit_parser for sandhi preprocessing
- **Text Processing**: fuzzywuzzy, python-Levenshtein, rapidfuzz
- **File Formats**: pysrt, pyyaml, chardet
- **Testing**: pytest with coverage
- **Development Tools**: black, flake8, mypy (optional - not extensively used)
- **Logging**: structlog
- **CLI**: click
- **Configuration**: YAML-based with environment variable overrides

## Current Status
- ✅ **Stories 2.1-2.3**: Complete (Sanskrit/Hindi identification, contextual modeling, scripture processing)
- 🚧 **Story 2.4.1**: Sanskrit sandhi preprocessing ready
- 📋 **Epic 2.4**: Research-grade hybrid matching pipeline
- 🚧 **Story 3.1**: NER module for Yoga Vedanta proper nouns
- 📋 **Story 3.2**: MCP integration for context-aware processing

## Architecture
MVP monolith with progressive enhancement through stories. Uses file-based data storage with JSON/YAML lexicons. Main entry point through src/main.py CLI with commands for single/batch processing.