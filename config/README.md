# Configuration Directory

This directory contains configuration files for the ASR Post-Processing system.

## Configuration Files

### Core Configuration
- `file_naming.yaml` - File naming patterns and validation rules
- `storage_management.yaml` - Directory structure and file processing rules

### Story 2.1: Lexicon-Based Correction System ✅
The system now supports comprehensive configuration for the lexicon-based correction system:

#### Sanskrit/Hindi Processing Configuration
```yaml
# Lexicon Management
lexicon_dir: "data/lexicons"
enable_lexicon_caching: true
english_words_file: null  # Optional English dictionary

# Fuzzy Matching Parameters
fuzzy_min_confidence: 0.75
levenshtein_threshold: 0.80
phonetic_threshold: 0.85
max_edit_distance: 3
enable_phonetic_matching: true
enable_compound_matching: true

# IAST Transliteration
iast_strict_mode: true

# Correction Application
correction_min_confidence: 0.80
correction_critical_confidence: 0.95
enable_context_validation: true
max_corrections_per_segment: 10
```

### Future Configuration Files
- `processing_pipeline.yaml` - Processing stage configuration (Epic 2)
- `quality_assurance.yaml` - QA metrics and thresholds (Epic 3)

## Default Configuration

The `SanskritPostProcessor` includes comprehensive default settings that work out-of-the-box:

- **Text Normalization**: Advanced filler word removal, number conversion
- **Conversational Pattern Detection**: Context-aware processing
- **Quality Validation**: Semantic drift and integrity checking
- **Metrics Collection**: Detailed processing statistics

## Configuration Override

You can override defaults by providing a configuration file:

```python
from pathlib import Path
from post_processors.sanskrit_post_processor import SanskritPostProcessor

config_path = Path("config/custom_config.yaml")
processor = SanskritPostProcessor(config_path=config_path)
```

## Usage

Configuration files use YAML format for easy editing by linguistic experts and system administrators.

## Environment Variables

Environment-specific overrides can be set using:
- `ASR_CONFIG_PATH` - Override default config directory
- `ASR_DATA_PATH` - Override default data directory
- `ASR_LOG_LEVEL` - Set logging level (DEBUG, INFO, WARN, ERROR)

## Validation

All configuration parameters are validated at startup with appropriate defaults and error handling for missing or invalid values.