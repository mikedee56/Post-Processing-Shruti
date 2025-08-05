# Configuration Directory

This directory contains configuration files for the ASR Post-Processing system.

## Configuration Files

### Core Configuration
- `file_naming.yaml` - File naming patterns and validation rules
- `storage_management.yaml` - Directory structure and file processing rules

### Future Configuration Files
- `processing_pipeline.yaml` - Processing stage configuration (Epic 2)
- `sanskrit_hindi_config.yaml` - Language identification settings (Epic 2)
- `transliteration_config.yaml` - IAST transliteration rules (Epic 2)
- `quality_assurance.yaml` - QA metrics and thresholds (Epic 3)

## Usage

Configuration files use YAML format for easy editing by linguistic experts and system administrators.

## Environment Variables

Environment-specific overrides can be set using:
- `ASR_CONFIG_PATH` - Override default config directory
- `ASR_DATA_PATH` - Override default data directory
- `ASR_LOG_LEVEL` - Set logging level (DEBUG, INFO, WARN, ERROR)