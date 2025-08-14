# Codebase Structure

## Directory Layout
```
/
├── src/                          # Main source code
│   ├── main.py                   # CLI entry point
│   ├── post_processors/          # Core processing modules
│   │   ├── sanskrit_post_processor.py  # Main processor
│   │   └── academic_polish_processor.py
│   ├── sanskrit_hindi_identifier/ # Language identification
│   ├── ner_module/              # Named Entity Recognition
│   ├── contextual_modeling/     # N-gram models, semantic similarity
│   ├── scripture_processing/    # Canonical verse handling
│   ├── utils/                   # Utility modules
│   │   ├── text_normalizer.py   # Basic text processing
│   │   ├── advanced_text_normalizer.py # MCP-enhanced processing
│   │   └── srt_parser.py        # SRT file handling
│   └── config/                  # Configuration management
├── tests/                       # Test suite
├── data/                        # Data files
│   ├── raw_srts/               # Input SRT files
│   ├── processed_srts/         # Output files
│   ├── lexicons/               # Sanskrit/Hindi dictionaries
│   └── golden_dataset/         # Benchmark data
├── config/                     # Configuration files
├── docs/                       # Documentation
└── scripts/                    # Helper scripts
```

## Core Components
- **SanskritPostProcessor**: Main processing engine with NER and academic polish capabilities
- **AdvancedTextNormalizer**: MCP-enhanced text processing with context classification
- **LexiconManager**: Sanskrit/Hindi dictionary management
- **SRTParser**: SRT file format handling with timestamp preservation
- **MetricsCollector**: Processing statistics and quality metrics

## Key Entry Points
- `src/main.py`: CLI with commands: process-single, process-batch, stats, validate
- Quick start scripts: `QUICK_START.bat`, `run_test.bat`
- Batch processing: `simple_batch.py`