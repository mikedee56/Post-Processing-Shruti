# Code Components Reference: Current System Analysis

## **Key Files for BMAD Review**

### **🔧 Core Processing Components**

#### **1. Text Normalization (Current System)**
**File**: `src/utils/text_normalizer.py`  
**Key Methods**:
```python
class TextNormalizer:
    def convert_numbers(self, text):
        # PROBLEMATIC METHOD - Line 156-162
        for word_num, digit in self.basic_numbers.items():
            pattern = rf'\b{re.escape(word_num)}\b'
            text = re.sub(pattern, digit, text, flags=re.IGNORECASE)
        return text
    
    def normalize_with_advanced_tracking(self, text):
        # Main normalization pipeline - Line 89-134
        # Calls convert_numbers() which causes issues
```

**Issue Location**: Lines 156-162 contain the problematic number conversion logic

#### **2. Sanskrit Post Processor (Main Pipeline)**
**File**: `src/post_processors/sanskrit_post_processor.py`  
**Integration Point**:
```python
class SanskritPostProcessor:
    def _process_srt_segment(self, segment, file_metrics):
        # Line 445-489 - Main processing pipeline
        # Calls text_normalizer.normalize_with_advanced_tracking()
        # This is where MCP integration would hook in
```

**Integration Opportunity**: Lines 460-470 where text normalization occurs

#### **3. Configuration Management**
**File**: `config/processing_config.yaml`
```yaml
# Current configuration structure
text_normalization:
  enable_number_conversion: true  # Controls problematic behavior
  enable_filler_removal: true
  enable_capitalization: true
```

**Enhancement Point**: Add MCP configuration section here

---

### **🧪 Test Components**

#### **1. Current Test Suite**
**File**: `tests/test_ner_module.py` (29 tests, 100% passing)
**Coverage**: Story 3.1 NER functionality comprehensive

#### **2. Text Normalization Tests**
**File**: `tests/test_text_normalizer.py`
**Key Test Cases**:
```python
def test_convert_numbers_basic():
    # Tests that currently pass but demonstrate the problem
    normalizer = TextNormalizer()
    result = normalizer.convert_numbers("chapter two verse twenty five")
    assert result == "chapter 2 verse 25"  # This works correctly
    
    # BUT this test doesn't exist (and would fail):
    # result = normalizer.convert_numbers("one by one")
    # assert result == "one by one"  # Should preserve idioms
```

**Missing Test Coverage**: Idiomatic expression preservation

---

### **📊 Sample Content Files**

#### **1. Problematic Content**
**File**: `c:\Users\miked\Downloads\engine_0_whisperx_large-v2_WhisperXEngine_large-v2 (6).srt`
**Key Problematic Segments**:
- Line 24: "And 1 by 1, he killed 6 of their children."
- Line 101: "He killed all the 6 children."
- Context: High-quality spiritual content degraded by processing

#### **2. Processed Output**
**File**: `data/processed_srts/engine_0_whisperx_large-v2_WhisperXEngine_large-v2_(6)_PROCESSED.srt`
**Demonstrates**: 
- ✅ Excellent Sanskrit processing ("Krishna" → "Kṛṣṇa")
- ❌ Poor number handling ("one by one" → "1 by 1")

---

### **🔧 Configuration Files**

#### **1. NER Configuration (Working Well)**
**File**: `config/ner_config.yaml`
```yaml
# Successful Story 3.1 implementation
entity_recognition:
  enable_ner: true
  confidence_threshold: 0.7
  categories:
    - PERSON
    - PLACE
    - TEXT
    - CONCEPT
```

#### **2. Proposed MCP Configuration**
**New File**: `config/advanced_normalization_config.yaml`
```yaml
# Proposed configuration for MCP integration
mcp_integration:
  enable_advanced_normalization: true
  mcp_servers:
    - name: "nlp-server"
      endpoint: "mcp://localhost:8000"
      capabilities: ["context_analysis", "pos_tagging"]
    - name: "spacy-server" 
      endpoint: "mcp://localhost:8001"
      capabilities: ["tokenization", "linguistic_analysis"]
  
  context_classification:
    confidence_threshold: 0.8
    preserve_patterns:
      idiomatic: ["one by one", "two by two", "step by step"]
      narrative: ["all the X", "one of the"]
    convert_patterns:
      mathematical: ["two thousand X", "X hundred"]
      scriptural: ["chapter X verse Y"]
  
  fallback_strategy:
    enable_enhanced_rules: true
    fallback_on_mcp_failure: true
```

---

### **📈 Data Models**

#### **1. Current Data Structures**
**File**: `src/utils/srt_parser.py`
```python
@dataclass
class SRTSegment:
    index: int
    start_time: str
    end_time: str
    text: str
    # Used throughout pipeline - no changes needed
```

#### **2. Proposed Enhancement Structures**
```python
@dataclass
class NumberContext:
    original_text: str
    context_type: str  # 'idiomatic', 'mathematical', 'narrative'
    confidence_score: float
    suggested_action: str  # 'preserve', 'convert', 'review'

@dataclass  
class AdvancedNormalizationResult:
    normalized_text: str
    contexts_identified: List[NumberContext]
    mcp_processing_time: float
    fallback_used: bool
```

---

### **🎯 Integration Architecture**

#### **Current Processing Flow**
```python
# Current flow in sanskrit_post_processor.py
def _process_srt_segment(self, segment, file_metrics):
    # 1. Basic preprocessing
    text = segment.text
    
    # 2. Text normalization (PROBLEM HERE)
    normalized_result = self.text_normalizer.normalize_with_advanced_tracking(text)
    text = normalized_result.corrected_text
    
    # 3. NER processing (WORKS GREAT)
    if self.enable_ner:
        text = self._apply_ner_corrections(text)
    
    # 4. Return processed segment
    return SRTSegment(segment.index, segment.start_time, segment.end_time, text)
```

#### **Proposed Enhanced Flow**
```python
# Enhanced flow with MCP integration
def _process_srt_segment(self, segment, file_metrics):
    # 1. Basic preprocessing
    text = segment.text
    
    # 2. ENHANCED normalization with MCP
    if self.config.get('enable_advanced_normalization', False):
        normalized_result = self.advanced_normalizer.normalize_with_context(text)
    else:
        normalized_result = self.text_normalizer.normalize_with_advanced_tracking(text)
    
    text = normalized_result.corrected_text
    
    # 3. NER processing (unchanged)
    if self.enable_ner:
        text = self._apply_ner_corrections(text)
    
    # 4. Return processed segment
    return SRTSegment(segment.index, segment.start_time, segment.end_time, text)
```

---

### **🚀 Development Entry Points**

#### **1. Primary Integration Point**
**File**: `src/utils/text_normalizer.py`  
**Method**: `convert_numbers()` at line 156
**Action**: Replace with context-aware logic

#### **2. Configuration Integration**
**File**: `src/post_processors/sanskrit_post_processor.py`
**Method**: `__init__()` around line 85
**Action**: Add MCP client initialization

#### **3. Testing Integration**  
**File**: `tests/test_advanced_number_normalization.py` (NEW)
**Content**: Comprehensive test suite for context-aware processing

#### **4. Pipeline Integration**
**File**: `src/post_processors/sanskrit_post_processor.py`
**Method**: `_process_srt_segment()` around line 460  
**Action**: Add conditional MCP processing path

---

### **📝 Quick Reference Commands**

#### **Test Current System**
```bash
# Run current tests
/c/Windows/py.exe -3.10 -m pytest tests/test_ner_module.py -v

# Process sample file
/c/Windows/py.exe -3.10 src/main.py process-single "input.srt" "output.srt"
```

#### **Examine Problem Areas**
```bash
# View problematic code
grep -n "convert_numbers" src/utils/text_normalizer.py

# Check configuration
cat config/processing_config.yaml
```

#### **Review Sample Content**
```bash
# See problematic output
grep -n "1 by 1" data/processed_srts/engine_0_*.srt
```

---

**Code Reference Prepared**: August 11, 2025  
**Technical Lead**: Quinn (Senior Developer & QA Architect)  
**Purpose**: Enable informed BMAD team technical discussion  
**Next Step**: Use this reference during architecture review session