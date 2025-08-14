# Suggested Commands for Development

## Most Common Development Commands

### Quick Setup and Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Set Python path
export PYTHONPATH="$PWD/src"

# Run comprehensive test suite
python3 -m pytest tests/ -v

# Quick functional test
python3 -c "import sys; sys.path.insert(0, 'src'); from post_processors.sanskrit_post_processor import SanskritPostProcessor; print('Import successful')"
```

### Primary Processing Commands
```bash
# Process a single SRT file
python3 src/main.py process-single data/test_samples/basic_test.srt output/test_result.srt

# Process with custom configuration
python3 src/main.py process-single input.srt output.srt --config config/academic_polish_config.yaml

# Get processing statistics
python3 src/main.py stats

# Validate processing quality
python3 src/main.py validate data/test_samples/basic_test.srt
```

### Testing Specific Components
```bash
# Test text normalization
python3 -m pytest tests/test_text_normalizer.py -v

# Test Sanskrit/Hindi identification
python3 -m pytest tests/test_sanskrit_hindi_correction.py -v

# Test academic polish functionality
python3 -m pytest tests/test_academic_polish_processor.py -v

# Test complete pipeline
python3 -m pytest tests/test_processing_pipeline.py -v
```

### Development Validation Commands
```bash
# Test MCP integration functionality
python3 -c "
import sys; sys.path.insert(0, 'src')
from utils.advanced_text_normalizer import AdvancedTextNormalizer
config = {'enable_mcp_processing': True}
normalizer = AdvancedTextNormalizer(config)
result = normalizer.convert_numbers_with_context('Chapter two verse twenty five')
print(f'Result: {result}')
"

# Test Sanskrit processing
python3 -c "
import sys; sys.path.insert(0, 'src')
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier
identifier = SanskritHindiIdentifier()
words = identifier.identify_words('Today we study yoga and dharma')
print(f'Identified {len(words)} Sanskrit/Hindi words')
"

# Quick processing validation
python3 -c "
import sys; sys.path.insert(0, 'src')
from post_processors.sanskrit_post_processor import SanskritPostProcessor
processor = SanskritPostProcessor()
print('✅ SanskritPostProcessor initialized successfully')
"
```

### Batch Operations
```bash
# Process all SRT files (Windows)
./QUICK_START.bat

# Simple batch processing
python3 simple_batch.py

# Process specific directory of SRT files
python3 src/main.py process-batch --input-dir data/raw_srts/ --output-dir data/processed_srts/
```

### Configuration and Setup
```bash
# Validate configuration files
python3 -c "import yaml; yaml.safe_load(open('config/academic_polish_config.yaml'))"

# Check available configuration options
ls config/*.yaml

# Test with different configurations
python3 src/main.py process-single input.srt output.srt --config config/ner_config.yaml
```

## Essential Commands for New Developers
1. `pip install -r requirements.txt` - Install all dependencies
2. `export PYTHONPATH="$PWD/src"` - Set up Python path
3. `python3 -m pytest tests/test_project_setup.py -v` - Validate setup
4. `python3 src/main.py process-single data/test_samples/basic_test.srt /tmp/test.srt` - Test basic functionality
5. `python3 -m pytest tests/ -v` - Run full test suite