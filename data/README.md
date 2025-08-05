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

### `lexicons/`
- **Purpose**: Externalized Sanskrit/Hindi dictionaries
- **File types**: JSON and YAML files
- **Version control**: Date-based versioning (`YYYY_MM_DD`)
- **Categories**:
  - Sanskrit terms
  - Hindi terms  
  - Vedanta proper nouns
  - Scriptural verses

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