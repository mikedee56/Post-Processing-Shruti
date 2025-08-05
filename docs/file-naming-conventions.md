# File Naming Conventions & Storage Management

**Story Reference**: 1.1.file-naming-conventions-storage-management  
**Status**: Implemented  
**Last Updated**: August 5, 2025

## Overview

This document defines the standardized file naming conventions and storage management approach for the Advanced ASR Post-Processing Workflow system. These conventions ensure consistent organization, easy retrieval, and scalable management of yoga vedanta lecture transcripts.

## Directory Structure

The system uses the following directory structure:

```
Post-Processing-Shruti/
├── data/
│   ├── raw_srts/          # Input SRT files (read-only)
│   ├── processed_srts/     # Output processed files
│   ├── lexicons/          # Externalized lexicon files
│   └── golden_dataset/    # Reference transcripts (read-only)
├── config/                # Configuration files
├── logs/                  # Processing and error logs (auto-created)
└── docs/                  # Documentation
```

## File Naming Patterns

### 1. SRT Files (Input)

**Pattern**: `{lecture_id}_{date}_{speaker}.srt`

**Examples**:
- `VED001_20241201_SwamiBrahmananda.srt`
- `VED142_20241215_AcharyaRamesh.srt`
- `VED003_20241203_SwamiBrahmavidyananda.srt`

**Components**:
- **lecture_id**: Format `VED{number:03d}` (e.g., VED001, VED002)
- **date**: Format `YYYYMMDD` (ISO date without separators)
- **speaker**: CamelCase format without spaces

### 2. Processed Files (Output)

**Pattern**: `{original_name}_processed_{version}.srt`

**Examples**:
- `VED001_20241201_SwamiBrahmananda_processed_v1.srt`
- `VED001_20241201_SwamiBrahmananda_processed_v2.srt`

**Versioning**: `v{number}` format for iterative improvements

### 3. Lexicon Files

**Sanskrit Terms**: `sanskrit_lexicon_{version}.json`  
**Hindi Terms**: `hindi_lexicon_{version}.json`  
**Proper Nouns**: `vedanta_proper_nouns_{version}.json`  
**Scriptural Verses**: `scriptural_verses_{version}.json`

**Version Format**: `{YYYY}_{MM}_{DD}`

**Examples**:
- `sanskrit_lexicon_2024_12_01.json`
- `vedanta_proper_nouns_2024_12_15.json`

### 4. Golden Dataset Files

**Pattern**: `{lecture_id}_{date}_{speaker}_golden.srt`

**Example**: `VED001_20241201_SwamiBrahmananda_golden.srt`

## Storage Management Rules

### File Organization

1. **Preserve Input Structure**: Output files maintain the same relative directory structure as input files
2. **Version Control**: Multiple versions of processed files are kept with version suffixes
3. **Read-Only Protection**: Original files in `raw_srts/` and `golden_dataset/` are protected

### File Validation

**SRT Files**:
- Maximum size: 50MB
- Required extension: `.srt`
- Encoding: UTF-8
- Minimum size: 100 bytes

**Lexicon Files**:
- Allowed extensions: `.json`, `.yaml`
- Encoding: UTF-8
- Version control enabled

**Naming Constraints**:
- Maximum filename length: 255 characters
- Forbidden characters: `< > : " | ? *`
- Case sensitive naming

### Processing Queue Management

- **Batch size**: 10 files per batch
- **Concurrent processing**: Maximum 3 files
- **Priority order**: 
  1. Golden dataset files
  2. High confidence files
  3. Standard files
- **Skip existing**: Processed files are not reprocessed unless forced

### Storage Limits

- **Total storage limit**: 100GB
- **Files per directory**: Maximum 1,000 files
- **Log retention**: 30 days
- **Processing history cleanup**: 90 days

## Configuration Files

The naming conventions are defined in:
- `config/file_naming.yaml` - Naming patterns and validation rules
- `config/storage_management.yaml` - Directory structure and processing rules

## File Discovery and Ingestion

### Discovery Patterns
- **Include**: `*.srt` files
- **Exclude**: `*_backup*`, `*_temp*`, hidden files (`.*)` 
- **Scan subdirectories**: Yes

### Validation Pipeline
1. Check file format validity
2. Validate SRT timestamps
3. Verify encoding (UTF-8)
4. Size validation (100 bytes - 50MB)
5. Quarantine invalid files to `data/quarantine/`

## Output Management

### Directory Mirroring
- Processed files maintain the same relative path structure as input files
- Missing output directories are created automatically
- Original file permissions are not preserved (use system defaults)

### Metadata Tracking
- Processing manifest created: `processing_manifest.json`
- File checksums included for integrity verification
- Processing time and quality metrics logged

## Integration Points

### Pre-Processing Hooks
1. Validate file format
2. Check available storage space
3. Create output directory structure

### Post-Processing Hooks
1. Verify output quality
2. Update processing manifest
3. Log completion status

## Error Handling

- **Invalid files**: Quarantined to `data/quarantine/`
- **Processing failures**: Retry up to 3 times with 60-second delays
- **Continue on error**: Single file failures don't stop batch processing
- **Error logging**: All validation and processing errors logged

## Usage Examples

### Valid File Names
✅ `VED001_20241201_SwamiBrahmananda.srt`  
✅ `VED142_20241215_AcharyaRamesh.srt`  
✅ `sanskrit_lexicon_2024_12_01.json`  

### Invalid File Names
❌ `lecture1.srt` (missing date and speaker)  
❌ `VED1_20241201_Swami Brahmananda.srt` (spaces in speaker name)  
❌ `VED001_12-01-2024_SwamiBrahmananda.srt` (incorrect date format)  

## Future Considerations

- **Date-based folders**: May be enabled for very large datasets
- **Speaker separation**: Directory organization by speaker if needed
- **Automatic archival**: For files older than retention periods
- **Cloud storage integration**: For backup and scalability

## Compliance Notes

- Follows IAST transliteration standards for Sanskrit terms
- Maintains academic integrity requirements
- Supports externalized lexicon management for linguistic experts
- Preserves original speech timestamps and content structure