# Development Commands

## Testing Commands
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_file_structure.py -v

# Run tests with coverage
python3 -m pytest tests/ --cov=src --cov-report=html

# Run specific test method
python3 -m pytest tests/test_srt_parser.py::TestSRTParser::test_parse_valid_srt_string -v
```

## Main Application Commands
```bash
# Process single SRT file
python3 src/main.py process-single input.srt output.srt

# Process single file with config
python3 src/main.py process-single input.srt output.srt --config config/custom.yaml

# Process batch of files
python3 src/main.py process-batch

# Get processing statistics
python3 src/main.py stats

# Validate processing quality
python3 src/main.py validate input.srt
```

## Quick Start Commands
```bash
# Windows quick start (installs deps and processes all SRTs)
./QUICK_START.bat

# Basic test run
./run_test.bat

# Simple batch processing
python3 simple_batch.py
```

## Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set Python path for testing
export PYTHONPATH="$PWD/src"

# Or on Windows
set PYTHONPATH=%~dp0src
```

## Environment Variables
- `ASR_CONFIG_PATH`: Override default config directory
- `ASR_DATA_PATH`: Override default data directory  
- `ASR_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARN, ERROR)

## Git Commands
Standard git workflow - no special git hooks or procedures identified.

## Linting/Formatting
- **black**: Code formatting (configured but not enforced)
- **flake8**: Code linting (configured but not enforced)
- **mypy**: Type checking (minimal usage, some files have ignore comments)