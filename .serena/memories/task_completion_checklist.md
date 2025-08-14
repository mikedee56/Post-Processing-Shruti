# Task Completion Checklist

## When Completing a Development Task

### 1. Code Quality Checks
```bash
# Run relevant tests
python3 -m pytest tests/test_[relevant_module].py -v

# Run all tests if significant changes
python3 -m pytest tests/ -v

# Check imports work correctly
python3 -c "import sys; sys.path.insert(0, 'src'); from [module] import [class]"
```

### 2. Integration Testing
```bash
# Test main CLI functionality
python3 src/main.py process-single [test_file.srt] [output.srt]

# Validate with test data
python3 src/main.py validate [test_file.srt]

# Check processing statistics
python3 src/main.py stats
```

### 3. Configuration Validation
- Ensure new features have appropriate config options in `config/` directory
- Test with both default and custom configurations
- Validate YAML configuration file syntax

### 4. Performance Verification
- Processing time should remain <2 seconds per file
- Memory usage should not significantly increase
- Test with representative SRT files from `data/test_samples/`

### 5. Documentation Updates
- Update docstrings for new/modified functions
- Update configuration README if new config options added
- Update relevant story files if implementing story requirements

### 6. Backward Compatibility
- Ensure existing API remains functional
- Test that existing processed files still work
- Validate lexicon compatibility

## Critical Validations

### For Text Processing Changes
```bash
# Test critical number conversion cases
python3 -c "
import sys; sys.path.insert(0, 'src')
from utils.text_normalizer import TextNormalizer
normalizer = TextNormalizer()
test_cases = ['Chapter two verse twenty five', 'Year two thousand five', 'And one by one']
for case in test_cases:
    result = normalizer.convert_numbers(case)
    print(f'{case} -> {result}')
"
```

### For Sanskrit Processing Changes
```bash
# Test Sanskrit term identification
python3 -c "
import sys; sys.path.insert(0, 'src')
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier
identifier = SanskritHindiIdentifier()
result = identifier.identify_words('Today we study yoga and dharma')
print(f'Identified {len(result)} words')
"
```

### For SRT Processing Changes
```bash
# Test end-to-end SRT processing
python3 src/main.py process-single data/test_samples/basic_test.srt /tmp/test_output.srt
# Verify output file was created and has valid SRT format
```

## No Build/Lint Commands
According to CLAUDE.md, there are no formal build/lint commands for this project. Testing is the primary validation method.