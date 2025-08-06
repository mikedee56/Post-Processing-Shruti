# Data Directory

This directory contains all data files for the ASR Post-Processing system.

## Directory Structure

### `raw_srts/`
- **Purpose**: Input SRT files from ASR system
- **Permissions**: Read-only (originals protected)
- **File format**: `.srt` files with UTF-8 encoding
- **Naming**: `{lecture_id}_{date}_{speaker}.srt`

### `processed_srts/`
- **Purpose**: Output processed files
- **Structure**: Mirrors input directory structure
- **Versioning**: Files include version suffixes (`_v1`, `_v2`, etc.)
- **Naming**: `{original_name}_processed_{version}.srt`

### `lexicons/` ✅ Enhanced (Story 2.1)
- **Purpose**: Externalized Sanskrit/Hindi dictionaries with advanced correction capabilities
- **File types**: JSON and YAML files with comprehensive validation
- **Version control**: Automated metadata tracking and checksum validation
- **Categories**:
  - `corrections.yaml` - Common term corrections and variations
  - `proper_nouns.yaml` - Deities, teachers, places with proper capitalization
  - `phrases.yaml` - Sanskrit/Hindi phrases and expressions
  - `verses.yaml` - Scriptural verse references with canonical text

#### Lexicon Entry Format (Story 2.1)
```yaml
entries:
  - original_term: "krishna"
    variations: ["krsna", "krshna", "krisna"] 
    transliteration: "Kṛṣṇa"
    is_proper_noun: true
    category: "deity"
    confidence: 1.0
    source_authority: "academic_standard"
```

#### Advanced Features
- **Fuzzy Matching**: Levenshtein distance and phonetic pattern matching
- **IAST Compliance**: Automatic transliteration standard enforcement  
- **Validation System**: Entry integrity checks and duplicate detection
- **Performance Optimization**: In-memory caching and indexed search structures

### `golden_dataset/`
- **Purpose**: Manually perfected reference transcripts
- **Permissions**: Read-only (reference data protected)
- **Usage**: Quality benchmarking and system validation
- **Naming**: `{lecture_id}_{date}_{speaker}_golden.srt`

## File Size Limits

- SRT files: Maximum 50MB each
- Total storage: 100GB limit
- Files per directory: Maximum 1,000 files

## Backup and Recovery

- Original files in `raw_srts/` and `golden_dataset/` are never modified
- Processed files maintain version history
- Lexicons are backed up on updates