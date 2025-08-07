# Sanskrit Sandhi Preprocessing Integration Guide

## Overview

The Sanskrit Sandhi Preprocessing system enhances the existing Story 2.1 lexicon-based correction system by splitting Sanskrit compound words (sandhi) before lexicon matching. This improves word identification accuracy for Sanskrit texts containing compound words.

## Architecture

### Components

#### SandhiPreprocessor (`src/sanskrit_hindi_identifier/sandhi_preprocessor.py`)
- **Purpose**: Core sandhi splitting functionality using `sanskrit_parser` library
- **Functionality**:
  - Identifies likely Sanskrit text requiring sandhi splitting
  - Splits compound words into constituent parts
  - Provides multiple segmentation candidates with confidence scores
  - Graceful fallback to basic tokenization when `sanskrit_parser` fails

#### Enhanced SanskritHindiIdentifier (`src/sanskrit_hindi_identifier/word_identifier.py`)
- **Purpose**: Integrates sandhi preprocessing into existing word identification pipeline
- **Enhancement**: Adds optional sandhi preprocessing step before lexicon matching
- **Compatibility**: Maintains full backward compatibility with Story 2.1 API

### Data Models

#### SandhiSplitCandidate
```python
@dataclass
class SandhiSplitCandidate:
    original_text: str                           # Original compound text
    segments: List[str]                          # Segmented parts
    confidence_score: float                      # Confidence (0.0-1.0)
    confidence_level: SegmentationConfidenceLevel # HIGH/MEDIUM/LOW/FALLBACK
    splitting_method: str                        # Method used for splitting
    metadata: Dict[str, Any]                     # Additional metadata
```

#### SandhiSplitResult
```python
@dataclass
class SandhiSplitResult:
    original_text: str                           # Original input text
    primary_candidate: SandhiSplitCandidate      # Best segmentation
    alternative_candidates: List[SandhiSplitCandidate] # Alternative segmentations
    preprocessing_successful: bool               # Processing succeeded
    fallback_used: bool                          # Used basic tokenization
    processing_time_ms: float                    # Processing time
```

## Integration Pattern

### Preprocessing Flow
```
Input Text → SandhiPreprocessor → Segmented Text → SanskritHindiIdentifier → Word Identification
```

### Processing Steps
1. **Input Validation**: Check for None/empty input
2. **Sanskrit Detection**: Determine if text likely contains Sanskrit
3. **Sandhi Splitting**: Use `sanskrit_parser` or heuristic methods
4. **Candidate Selection**: Choose best segmentation based on confidence
5. **Fallback Handling**: Use basic tokenization if processing fails
6. **Word Identification**: Process segmented text through existing pipeline

## Configuration

### Feature Flag
```python
# Enable sandhi preprocessing (default: True)
identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)

# Disable sandhi preprocessing for backward compatibility
identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=False)
```

### Runtime Control
```python
# Toggle sandhi preprocessing at runtime
identifier.set_sandhi_preprocessing_enabled(False)
identifier.set_sandhi_preprocessing_enabled(True)
```

## Usage Examples

### Basic Usage
```python
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier

# Initialize with sandhi preprocessing
identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)

# Process text (sandhi preprocessing applied automatically)
text = "yogaścittavṛttinirodhaḥ teaches us about consciousness"
identified_words = identifier.identify_words(text)

# Get sandhi preprocessing statistics
stats = identifier.get_sandhi_preprocessing_stats()
print(f"Processed {stats['total_processed']} texts")
print(f"Success rate: {stats['success_rate']:.1f}%")
```

### Direct Sandhi Preprocessing
```python
from sanskrit_hindi_identifier.sandhi_preprocessor import SandhiPreprocessor

# Initialize preprocessor
preprocessor = SandhiPreprocessor(enable_sandhi_preprocessing=True)

# Process Sanskrit compound
compound = "yogaścittavṛttinirodhaḥ"
result = preprocessor.preprocess_text(compound)

print(f"Original: {result.original_text}")
print(f"Segments: {result.primary_candidate.segments}")
print(f"Confidence: {result.primary_candidate.confidence_score}")
print(f"Method: {result.primary_candidate.splitting_method}")
```

## Performance Characteristics

### Processing Time
- **Target**: < 2x processing time increase vs non-sandhi processing
- **Typical**: 10-50ms per text segment
- **Fallback**: < 1ms when using basic tokenization

### Memory Usage
- **Minimal impact**: Stateless processing
- **Statistics tracking**: Small memory footprint for counters
- **No caching**: Each processing operation is independent

## Error Handling

### Graceful Degradation
1. **Sanskrit Parser Unavailable**: Falls back to basic tokenization
2. **Processing Errors**: Returns original text with fallback segmentation
3. **Invalid Input**: Handles None, empty, and malformed text gracefully
4. **Unicode Issues**: Processes mixed scripts and special characters

### Error Recovery
```python
# Check if sanskrit_parser is available
validation = identifier.validate_sandhi_preprocessing_config()
if validation['warnings']:
    print("Warnings:", validation['warnings'])
    print("Recommendations:", validation['recommendations'])
```

## Monitoring and Diagnostics

### Statistics
```python
stats = identifier.get_sandhi_preprocessing_stats()
# Returns:
# {
#   'total_processed': 100,
#   'successful_splits': 75,
#   'fallback_used': 25,
#   'processing_errors': 0,
#   'sanskrit_parser_available': True,
#   'preprocessing_enabled': True,
#   'success_rate': 75.0
# }
```

### Configuration Validation
```python
validation = identifier.validate_sandhi_preprocessing_config()
# Returns validation status, warnings, and recommendations
```

## Migration from Story 2.1

### Existing Code Compatibility
**No changes required** - existing code continues to work unchanged:

```python
# This existing code works without modification
identifier = SanskritHindiIdentifier()
words = identifier.identify_words("some text")
```

### Enabling Sandhi Preprocessing
```python
# Simply add the enable flag to get enhanced functionality
identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)
words = identifier.identify_words("yogaścittavṛtti")  # Now benefits from sandhi splitting
```

### Performance Considerations
- Monitor processing time impact using statistics
- Disable if performance requirements not met: `enable_sandhi_preprocessing=False`
- Use runtime toggling for A/B testing scenarios

## Dependencies

### Required
- `sanskrit_parser>=0.1.0` - Sanskrit sandhi analysis (graceful fallback if unavailable)
- Existing Story 2.1 dependencies (pandas, pyyaml, etc.)

### Optional
- Enhanced `sanskrit_parser` modules for improved accuracy (future enhancement)

## Testing

### Test Coverage
- **Unit Tests**: `tests/test_sandhi_preprocessing.py`
- **Integration Tests**: SanskritHindiIdentifier enhancement tests
- **Performance Tests**: Processing time benchmarks
- **Edge Case Tests**: Error handling and malformed input

### Running Tests
```bash
# Run all sandhi preprocessing tests
python -m pytest tests/test_sandhi_preprocessing.py -v

# Run performance benchmarks
python -m pytest tests/test_sandhi_preprocessing.py::TestSandhiPerformanceBenchmarks -v
```

## Future Story 2.4 Integration

### Preparation for Hybrid Matching
The sandhi preprocessing system is designed to integrate seamlessly with the planned Story 2.4 hybrid matching pipeline:

1. **Component Interface**: Standard interfaces ready for hybrid system integration
2. **Confidence Scoring**: Provides confidence metrics for hybrid decision making  
3. **Alternative Candidates**: Multiple segmentation options for advanced matching
4. **Performance Metrics**: Detailed statistics for hybrid system optimization

### Extension Points
- **Custom Splitting Methods**: Add domain-specific sandhi rules
- **Machine Learning Integration**: Incorporate ML-based segmentation models
- **Caching Layer**: Add segmentation result caching for performance
- **Batch Processing**: Optimize for large-scale text processing

## Troubleshooting

### Common Issues

#### Sanskrit Parser Import Errors
```
WARNING: sanskrit_parser library not available: cannot import name 'sandhi'
```
**Solution**: System automatically falls back to basic tokenization. Install correct version or accept fallback behavior.

#### Performance Degradation
**Symptoms**: Processing time > 2x baseline
**Solutions**: 
- Disable sandhi preprocessing: `enable_sandhi_preprocessing=False`
- Check for very long input texts causing performance issues
- Monitor statistics for high fallback usage indicating configuration issues

#### No Segmentation Results
**Symptoms**: All results use fallback tokenization
**Cause**: Text not detected as Sanskrit, or `sanskrit_parser` unavailable
**Solution**: Verify input contains Sanskrit text, check `sanskrit_parser` installation

## API Reference

See component docstrings in:
- `src/sanskrit_hindi_identifier/sandhi_preprocessor.py`
- `src/sanskrit_hindi_identifier/word_identifier.py`

## Version History

- **v1.0** (Story 2.4.1): Initial implementation with `sanskrit_parser` integration
- **Future** (Story 2.4): Enhanced hybrid matching integration