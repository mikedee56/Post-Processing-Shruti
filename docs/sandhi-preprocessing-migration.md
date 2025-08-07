# Sanskrit Sandhi Preprocessing Migration Guide

## Overview

This guide helps you migrate from Story 2.1 to Story 2.4.1 with Sanskrit sandhi preprocessing support. The migration is **backward compatible** - existing code continues to work without changes.

## Migration Scenarios

### Scenario 1: Continue Using Existing Functionality (No Changes)

**Use Case**: You want to keep current behavior unchanged.

**Migration**: **No changes required**

```python
# Existing code - continues to work exactly the same
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier

identifier = SanskritHindiIdentifier()
words = identifier.identify_words("your sanskrit text")
```

**Result**: 
- ✅ All existing functionality preserved
- ✅ Same performance characteristics
- ✅ Same output format
- ✅ No sandhi preprocessing applied

---

### Scenario 2: Enable Sandhi Preprocessing for New Projects

**Use Case**: You want to benefit from improved Sanskrit compound word identification.

**Migration**: **Add one parameter**

```python
# Enhanced code - enables sandhi preprocessing
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier

identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)
words = identifier.identify_words("yogaścittavṛttinirodhaḥ")
```

**Benefits**:
- ✅ Improved identification of Sanskrit compound words
- ✅ Better lexicon matching accuracy
- ✅ Graceful fallback when sandhi processing fails
- ✅ Same API and output format

---

### Scenario 3: Gradual Migration with Runtime Control

**Use Case**: You want to test sandhi preprocessing gradually or toggle it based on conditions.

**Migration**: **Use runtime control methods**

```python
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier

# Initialize with sandhi preprocessing enabled
identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)

# Process some texts with sandhi preprocessing
for text in sanskrit_texts:
    words = identifier.identify_words(text)
    
# Disable for performance-critical section
identifier.set_sandhi_preprocessing_enabled(False)
for text in large_text_batch:
    words = identifier.identify_words(text)  # Uses basic tokenization

# Re-enable for accuracy-critical section
identifier.set_sandhi_preprocessing_enabled(True)
```

**Benefits**:
- ✅ Flexible control over when sandhi preprocessing is used
- ✅ Performance tuning capabilities
- ✅ A/B testing support

---

## Feature Comparison

| Feature | Story 2.1 | Story 2.4.1 (Disabled) | Story 2.4.1 (Enabled) |
|---------|-----------|------------------------|------------------------|
| Word Identification | ✅ | ✅ | ✅ |
| Lexicon Matching | ✅ | ✅ | ✅ Enhanced |
| Sanskrit Compound Splitting | ❌ | ❌ | ✅ |
| Performance | Baseline | Baseline | < 2x Baseline |
| API Compatibility | ✅ | ✅ | ✅ |
| Dependencies | Minimal | Minimal | + sanskrit_parser |

## Performance Considerations

### Processing Time Impact

```python
import time
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier

# Benchmark both configurations
def benchmark_configuration(enable_sandhi: bool, test_texts: list):
    identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=enable_sandhi)
    
    start_time = time.time()
    for text in test_texts:
        identifier.identify_words(text)
    end_time = time.time()
    
    return (end_time - start_time) * 1000  # Return milliseconds

# Compare performance
test_texts = ["yogaścittavṛtti", "bhagavadgītā", "sarvadharmān"]

time_without_sandhi = benchmark_configuration(False, test_texts)
time_with_sandhi = benchmark_configuration(True, test_texts)

print(f"Without sandhi: {time_without_sandhi:.1f}ms")
print(f"With sandhi: {time_with_sandhi:.1f}ms")
print(f"Performance impact: {time_with_sandhi/time_without_sandhi:.1f}x")
```

### Performance Guidelines

- **Target**: < 2x processing time increase
- **Typical**: 1.2x - 1.5x for most Sanskrit texts
- **Fallback**: ~1x when `sanskrit_parser` uses basic tokenization
- **Monitor**: Use statistics to track performance impact

## Monitoring and Validation

### Statistics Monitoring

```python
identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)

# Process your texts
for text in your_texts:
    identifier.identify_words(text)

# Monitor sandhi preprocessing performance
stats = identifier.get_sandhi_preprocessing_stats()
print(f"Total processed: {stats['total_processed']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Fallback usage: {stats['fallback_used']}")
print(f"Sanskrit parser available: {stats['sanskrit_parser_available']}")

# Reset statistics for next batch
identifier.reset_sandhi_preprocessing_stats()
```

### Configuration Validation

```python
# Validate your configuration
validation = identifier.validate_sandhi_preprocessing_config()

if not validation['is_valid']:
    print("Configuration issues found:")
    for error in validation['errors']:
        print(f"  ERROR: {error}")

for warning in validation['warnings']:
    print(f"  WARNING: {warning}")

for recommendation in validation['recommendations']:
    print(f"  RECOMMENDATION: {recommendation}")
```

## Common Migration Patterns

### Pattern 1: Feature Flag Configuration

```python
import os
from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier

# Use environment variable to control sandhi preprocessing
ENABLE_SANDHI = os.getenv('ENABLE_SANDHI_PREPROCESSING', 'false').lower() == 'true'

identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=ENABLE_SANDHI)
```

### Pattern 2: Conditional Based on Text Content

```python
def create_identifier(text: str) -> SanskritHindiIdentifier:
    """Create identifier with sandhi preprocessing for Sanskrit-heavy text."""
    
    # Simple heuristic: enable sandhi if text contains Sanskrit indicators
    has_sanskrit_chars = any(c in text for c in 'āīūṛḥṃśṅṇṭḍṅ')
    has_long_words = any(len(word) > 10 for word in text.split())
    
    enable_sandhi = has_sanskrit_chars or has_long_words
    
    return SanskritHindiIdentifier(enable_sandhi_preprocessing=enable_sandhi)
```

### Pattern 3: Performance-Based Dynamic Switching

```python
class AdaptiveIdentifier:
    def __init__(self):
        self.identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)
        self.performance_threshold_ms = 100  # 100ms threshold
    
    def identify_words(self, text: str):
        import time
        
        start_time = time.time()
        result = self.identifier.identify_words(text)
        processing_time = (time.time() - start_time) * 1000
        
        # Disable sandhi if processing is too slow
        if processing_time > self.performance_threshold_ms:
            print(f"Slow processing detected ({processing_time:.1f}ms), disabling sandhi")
            self.identifier.set_sandhi_preprocessing_enabled(False)
        
        return result
```

## Dependency Management

### Requirements.txt Updates

The `sanskrit_parser` dependency is already added to requirements.txt:

```
# Story 2.4 Research Enhancement Dependencies
sanskrit_parser>=0.1.0  # Sanskrit sandhi splitting - Story 2.4.1 dependency
```

### Installation

```bash
# Install/update dependencies
pip install -r requirements.txt

# Or install specific dependency
pip install sanskrit_parser>=0.1.0
```

### Dependency Handling

The system handles missing dependencies gracefully:

```python
# This code works whether sanskrit_parser is installed or not
identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)

# Check if dependency is available
validation = identifier.validate_sandhi_preprocessing_config()
if 'sanskrit_parser library not available' in validation['warnings']:
    print("Using fallback tokenization - consider installing sanskrit_parser")
```

## Testing Your Migration

### Basic Functionality Test

```python
def test_migration():
    """Test that migration preserves existing functionality."""
    
    test_text = "yoga dharma consciousness"
    
    # Test Story 2.1 behavior (disabled sandhi)
    identifier_old = SanskritHindiIdentifier(enable_sandhi_preprocessing=False)
    result_old = identifier_old.identify_words(test_text)
    
    # Test Story 2.4.1 behavior (enabled sandhi)
    identifier_new = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)
    result_new = identifier_new.identify_words(test_text)
    
    # Both should return IdentifiedWord objects
    assert all(hasattr(word, 'word') for word in result_old)
    assert all(hasattr(word, 'word') for word in result_new)
    
    print("✅ Migration test passed - basic functionality preserved")

test_migration()
```

### Performance Test

```python
def test_performance():
    """Test that performance impact is within acceptable limits."""
    
    test_texts = ["yogaścittavṛtti"] * 100  # 100 Sanskrit texts
    
    # Benchmark both configurations
    import time
    
    # Without sandhi
    identifier_old = SanskritHindiIdentifier(enable_sandhi_preprocessing=False)
    start = time.time()
    for text in test_texts:
        identifier_old.identify_words(text)
    time_old = (time.time() - start) * 1000
    
    # With sandhi
    identifier_new = SanskritHindiIdentifier(enable_sandhi_preprocessing=True)
    start = time.time()
    for text in test_texts:
        identifier_new.identify_words(text)
    time_new = (time.time() - start) * 1000
    
    performance_ratio = time_new / time_old
    print(f"Performance impact: {performance_ratio:.2f}x")
    
    # Story requirement: < 2x processing time
    assert performance_ratio < 2.0, f"Performance impact too high: {performance_ratio:.2f}x"
    
    print("✅ Performance test passed - within 2x requirement")

test_performance()
```

## Rollback Plan

If you need to rollback to Story 2.1 behavior:

### Option 1: Disable Feature Flag

```python
# Disable sandhi preprocessing - returns to Story 2.1 behavior
identifier = SanskritHindiIdentifier(enable_sandhi_preprocessing=False)
```

### Option 2: Use Original Story 2.1 Code

Since the enhancement is backward compatible, your original Story 2.1 code continues to work without any changes.

### Option 3: Runtime Disable

```python
# Disable at runtime
identifier.set_sandhi_preprocessing_enabled(False)
```

## Support and Troubleshooting

### Common Issues

#### Issue: "sanskrit_parser library not available"
**Solution**: Either install `sanskrit_parser` or accept fallback behavior:
```bash
pip install sanskrit_parser>=0.1.0
```

#### Issue: Performance too slow
**Solution**: Disable sandhi preprocessing or tune performance:
```python
identifier.set_sandhi_preprocessing_enabled(False)
```

#### Issue: Unexpected segmentation results
**Solution**: Check input text and validate configuration:
```python
validation = identifier.validate_sandhi_preprocessing_config()
stats = identifier.get_sandhi_preprocessing_stats()
```

### Getting Help

1. **Check Statistics**: Use monitoring methods to understand behavior
2. **Validate Configuration**: Use validation methods to check setup
3. **Review Logs**: Enable debug logging for detailed processing information
4. **Test with Fallback**: Disable sandhi to isolate issues

## Next Steps

After successful migration:

1. **Monitor Performance**: Track processing times and adjust as needed
2. **Collect Metrics**: Use statistics to understand sandhi preprocessing benefits
3. **Prepare for Story 2.4**: The sandhi preprocessing is designed to integrate with future hybrid matching
4. **Provide Feedback**: Report any issues or improvements for future enhancements