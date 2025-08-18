# DEV HANDOFF: MCP-First Configuration Implementation

## 🎯 **IMMEDIATE TASK PRIORITY**
**User Responsibility**: Implement MCP-first configuration to complete architectural optimization

## 📍 **CONTEXT & STATUS**
- **Architectural Fix COMPLETED**: Eliminated duplicate NER initialization (2x memory overhead removed)
- **Performance Baseline**: 6.6 seg/sec average (need 51% improvement to reach 10+ seg/sec target)
- **MCP Libraries**: Successfully installed and validated
- **Critical Requirement**: MCP must be primary path, fallback only for genuine server failures

## 🔧 **EXACT CODE CHANGES REQUIRED**

### **File**: `src/utils/advanced_text_normalizer.py`
### **Lines**: 150-156 (approximately)

**CURRENT CODE (Fallback-Eager):**
```python
try:
    from mcp import client as mcp_client
    self.mcp_enabled = True
except ImportError:
    self.mcp_enabled = False  # Falls back immediately on import failure
```

**REQUIRED CODE (MCP-First):**
```python
try:
    from mcp import client as mcp_client
    if self._validate_mcp_connectivity():
        self.mcp_enabled = True
        self.logger.info("MCP server validated - using MCP processing")
    else:
        self.mcp_enabled = False
        self.logger.warning("MCP server unreachable - using fallback processing")
except ImportError:
    self.mcp_enabled = False
    self.logger.error("MCP libraries not available - using fallback processing")
```

### **REQUIRED NEW METHOD**
Add this method to the `AdvancedTextNormalizer` class:

```python
def _validate_mcp_connectivity(self) -> bool:
    """
    Validate MCP server connectivity before enabling MCP processing.
    Returns True if MCP server is reachable and functional.
    """
    try:
        # Test basic MCP server connection
        # Replace with actual MCP server validation logic
        response = self.mcp_client.ping()  # or equivalent health check
        return response.status == "healthy"
    except Exception as e:
        self.logger.warning(f"MCP server validation failed: {e}")
        return False
```

## 🧪 **VALIDATION REQUIREMENTS**

### **Test Command After Implementation:**
```bash
# Activate virtual environment
source .venv/bin/activate  # Linux
# OR
.venv\Scripts\activate  # Windows

# Set Python path
export PYTHONPATH=/mnt/d/Post-Processing-Shruti/src

# Test MCP-first behavior
python -c "
import sys
sys.path.insert(0, 'src')
from utils.advanced_text_normalizer import AdvancedTextNormalizer

config = {'enable_mcp_processing': True, 'enable_fallback': True}
normalizer = AdvancedTextNormalizer(config)

# Test critical cases
test_cases = [
    'And one by one, he killed six of their children.',  # Should preserve idiomatic
    'Chapter two verse twenty five.',                     # Should convert to digits
    'Year two thousand five.',                           # Critical bug fix test
]

for text in test_cases:
    result = normalizer.convert_numbers_with_context(text)
    print(f'Input: {text}')
    print(f'Output: {result}')
    print()
"
```

### **Expected Results:**
1. **Idiomatic Preservation**: "one by one" should remain unchanged
2. **Scriptural Conversion**: "Chapter 2 verse 25" with proper capitalization
3. **Year Fix**: "Year 2005" (critical bug validation)
4. **Logging**: Clear indication of MCP vs fallback usage

## 📊 **PERFORMANCE VALIDATION**

### **Performance Test Script:**
```bash
python -c "
import sys
import time
sys.path.insert(0, 'src')
from post_processors.sanskrit_post_processor import SanskritPostProcessor
from utils.srt_parser import SRTSegment

processor = SanskritPostProcessor()
test_segments = []

# Create 20 test segments
for i in range(1, 21):
    segment = SRTSegment(
        index=i,
        start_time=float(i),
        end_time=float(i+4),
        text=f'Today we study yoga and dharma segment {i}.',
        raw_text=f'Today we study yoga and dharma segment {i}.'
    )
    test_segments.append(segment)

# Measure performance
start_time = time.time()
for segment in test_segments:
    processor._process_srt_segment(segment, processor.metrics_collector.create_file_metrics('test'))
end_time = time.time()

total_time = end_time - start_time
segments_per_second = len(test_segments) / total_time

print(f'Performance: {segments_per_second:.2f} segments/sec')
print(f'Target: 10.0 segments/sec')
print(f'Status: {'TARGET ACHIEVED' if segments_per_second >= 10.0 else 'NEEDS IMPROVEMENT'}')
"
```

## 🚨 **CRITICAL SUCCESS CRITERIA**

### **MCP Configuration Success:**
- [ ] MCP-first logic implemented (not fallback-eager)
- [ ] `_validate_mcp_connectivity()` method added
- [ ] Proper logging for MCP vs fallback usage
- [ ] No ImportError fallback unless MCP genuinely unavailable

### **Performance Success:**
- [ ] Post-fix performance ≥ 10.0 segments/second
- [ ] Combined architectural + MCP improvements validated
- [ ] 51% improvement over baseline (6.6 → 10+ seg/sec)

### **Functional Success:**
- [ ] Idiomatic phrases preserved ("one by one")
- [ ] Scriptural references converted ("Chapter 2 verse 25")
- [ ] Year conversion bug fixed ("Year 2005")

## ⚡ **KNOWN CRITICAL ISSUES (Parallel Resolution)**

### **Issue 1: IndicNLP Classification Failures**
- **Impact**: 160-200% error rate on Sanskrit terms
- **Symptom**: Terms classified as "OTHER" instead of Sanskrit/Hindi
- **Timeline**: 3-5 days effort (parallel to MCP implementation)

### **Issue 2: Unicode Console Output**
- **Impact**: `UnicodeEncodeError` with IAST characters
- **Solution**: Environment variable configuration
- **Timeline**: Quick fix (hours)

## 📋 **POST-IMPLEMENTATION CHECKLIST**

1. **Code Implementation**:
   - [ ] MCP-first logic in `advanced_text_normalizer.py:150-156`
   - [ ] `_validate_mcp_connectivity()` method added
   - [ ] Logging statements added

2. **Testing**:
   - [ ] MCP vs fallback behavior validated
   - [ ] Performance test shows ≥10 seg/sec
   - [ ] Critical text cases pass validation

3. **Documentation**:
   - [ ] Update configuration documentation
   - [ ] Log MCP connectivity status
   - [ ] Performance metrics recorded

## 🔄 **NEXT STEPS AFTER COMPLETION**
1. Report performance results to PM/Architect
2. Address remaining critical issues (IndicNLP, Unicode)
3. Proceed with Epic 4 ($185K) go/no-go decision
4. Production readiness assessment

## 📞 **ESCALATION CONTACTS**
- **Technical Issues**: Architect (/BMad:agents:architect)
- **Project Management**: PM (/pm)
- **Performance Questions**: Review performance validation scripts above

---
**Last Updated**: 2025-08-16
**Priority**: CRITICAL - Blocks $235K investment decision
**Expected Completion**: 24-48 hours