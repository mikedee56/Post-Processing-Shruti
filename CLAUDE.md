# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Advanced ASR Post-Processing Workflow** project designed to transform ASR-generated transcripts of Yoga Vedanta lectures into highly accurate, academically rigorous textual resources. The system processes SRT files to correct Sanskrit/Hindi terms, apply IAST transliteration standards, and identify scriptural verses.

## Technology Stack

- **Language**: Python 3.10
- **Data Processing**: pandas
- **NLP Libraries**: iNLTK, IndicNLP Library for Indic language support
- **Specialized Model**: ByT5-Sanskrit (optional for advanced corrections)
- **Data Storage**: File-based approach with JSON/YAML lexicons
- **Version Control**: Git

## Project Structure

The project follows a monorepo structure with these key directories:

- `docs/` - Project documentation (PRD, architecture, tech stack)
- `data/` - Raw and processed transcript data
  - `raw_srts/` - Original SRT files
  - `processed_srts/` - Post-processed outputs
  - `lexicons/` - Externalized Sanskrit/Hindi dictionaries
  - `golden_dataset/` - Manually perfected transcripts for benchmarking
- `src/` - Source code (to be created)
  - `post_processors/` - Core post-processing modules
  - `sanskrit_hindi_identifier/` - Language identification logic
  - `ner_module/` - Named Entity Recognition
  - `qa_module/` - Quality assurance metrics
- `tests/` - Test suite
- `config/` - Configuration files
- `scripts/` - Helper scripts

## Key Requirements

### Functional Requirements
- Process SRT files while maintaining timestamp integrity
- Convert spoken numbers to digits ("two thousand five" → "2005")
- Remove filler words ("um", "uh")
- Apply IAST transliteration standard to Sanskrit/Hindi terms
- Identify and correct scriptural verses using canonical text
- Capitalize proper nouns specific to Yoga Vedanta

### Non-Functional Requirements
- Handle large data volumes (12,000+ hours of audio)
- Preserve original speech intention, tone, and style
- Support externalized lexicons for easy updates by linguistic experts
- Maintain scalability for future growth

## Development Approach

This is a **MVP monolith** designed for progressive complexity:
1. **Epic 1**: Foundation & pre-processing pipeline
2. **Epic 2**: Sanskrit & Hindi identification & correction
3. **Epic 3**: Semantic refinement & QA framework
4. **Epic 4**: Deployment & scalability

## Key Data Models

### Transcript Segment
- `text` (string): Transcribed text
- `start_time`/`end_time` (float): Timestamps in seconds
- `confidence_score` (float): ASR confidence
- `is_flagged` (boolean): Requires human review
- `correction_history` (array): Log of human corrections

### Lexicon Entry
- `original_term` (string): Correct term
- `variations` (array): Common misrecognized variations
- `transliteration` (string): IAST transliteration
- `is_proper_noun`/`is_verse` (boolean): Classification flags
- `canonical_text` (string): Full scriptural text for verses

## Testing Requirements

Follow a **Full Testing Pyramid** approach:
- Unit tests for individual functions
- Integration tests for the complete pipeline
- Use golden dataset for accuracy measurements (WER/CER reduction)

## Important Notes

- **No build/lint commands available** - this is a planning-stage project
- The existing `sanskrit_post_processor.py` script needs to be integrated
- Focus on defensive security - this processes academic content only
- Maintain academic integrity and IAST transliteration standards
- All external lexicons should be version-controlled JSON/YAML files