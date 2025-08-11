# Current System Analysis: Technical Architecture Review

## Current Text Normalization Implementation

### **File**: `src/utils/text_normalizer.py`

**Current Architecture**:
```python
class TextNormalizer:
    def __init__(self):
        self.basic_numbers = {
            "one": "1", "two": "2", "three": "3", "four": "4", 
            "five": "5", "six": "6", "seven": "7", "eight": "8"
        }
        
    def convert_numbers(self, text):
        # PROBLEMATIC: Blind find-and-replace without context
        for word_num, digit in self.basic_numbers.items():
            pattern = rf'\b{re.escape(word_num)}\b'
            text = re.sub(pattern, digit, text, flags=re.IGNORECASE)
        return text
```

---

## **✅ Strengths of Current System**

### **1. Epic 2.4 + Story 3.1 Success**
- **Sanskrit/Hindi Processing**: Excellent NER with IAST transliteration
- **Entity Recognition**: 100% success rate for proper nouns  
- **Lexicon Integration**: Sophisticated fuzzy matching and correction
- **Performance**: Processing under 2s target consistently

### **2. Robust Foundation**
- **Processing Pipeline**: Well-structured multi-stage architecture
- **Configuration Management**: Comprehensive YAML-based settings
- **Test Coverage**: 29 test cases with 100% pass rate
- **Error Handling**: Graceful fallbacks and logging

### **3. Production Quality**
- **Integration**: Seamless with existing SanskritPostProcessor
- **Scalability**: Handles large volume processing efficiently
- **Monitoring**: Comprehensive metrics and reporting system

---

## **❌ Critical Limitations Identified**

### **1. Primitive Number Processing**
```python
# Current approach: Context-blind replacement
def convert_numbers(self, text):
    for word_num, digit in self.basic_numbers.items():
        text = re.sub(rf'\b{word_num}\b', digit, text, flags=re.IGNORECASE)
    return text

# Result: "one by one" → "1 by 1" ❌
# Should be: Context-aware preservation of idioms ✅
```

### **2. No Linguistic Intelligence**
- **Zero Context Analysis**: Cannot distinguish mathematical vs idiomatic usage
- **No Semantic Understanding**: Treats all number words identically  
- **Missing POS Tagging**: No grammatical role recognition
- **No Phrase Recognition**: Cannot identify compound expressions

### **3. Rule-Based Limitations**
```python
# Current pattern matching is insufficient:
CONTEXT_PATTERNS = {
    'mathematical': [r'\btwo thousand \d+\b'],  # Limited coverage
    'idiomatic': []  # MISSING - This is the core problem
}
```

---

## **🔧 Integration Points for MCP Enhancement**

### **Current Integration Opportunities**

**1. TextNormalizer Enhancement**
```python
# Current: src/utils/text_normalizer.py:convert_numbers()
# Proposed: Enhanced with MCP context analysis
class AdvancedTextNormalizer(TextNormalizer):
    def __init__(self):
        super().__init__()
        self.mcp_client = MCPClient("nlp-server")  # NEW
        
    def convert_numbers(self, text):
        # NEW: Context-aware processing with MCP
        return self.mcp_client.process_numbers_with_context(text)
```

**2. Configuration Extension**
```yaml
# Current: config/processing_config.yaml
# Add new section:
advanced_normalization:
  enable_mcp_processing: true
  mcp_servers:
    - nlp-server
    - spacy-server
  context_classification:
    confidence_threshold: 0.8
    fallback_to_basic: true
```

**3. Pipeline Integration**
```python
# Current: SanskritPostProcessor integration point
def _apply_text_normalization(self, text):
    if self.config.get('enable_advanced_normalization', False):
        return self.advanced_normalizer.normalize_with_context(text)
    else:
        return self.text_normalizer.normalize_with_tracking(text)
```

---

## **📊 Performance Analysis**

### **Current System Metrics**
- **Processing Speed**: <2s per file (excellent)
- **Memory Usage**: Minimal footprint
- **Error Rate**: <1% with graceful fallbacks
- **Integration Complexity**: Low - well-abstracted interfaces

### **MCP Enhancement Impact Estimates**
- **Processing Speed**: 1.5-2x slower (still under targets)  
- **Memory Usage**: +30-50% (acceptable for quality improvement)
- **Error Rate**: Reduced overall due to better context handling
- **Integration Complexity**: Moderate - requires MCP client libraries

---

## **🎯 Technical Requirements for MCP Integration**

### **Infrastructure Needs**
1. **MCP Client Libraries**: Python MCP SDK installation
2. **External Dependencies**: SpaCy models, transformers packages
3. **Configuration Management**: Enhanced YAML schemas
4. **Testing Environment**: MCP server simulation capabilities

### **Development Requirements**
1. **New Modules**: `src/utils/mcp_number_processor.py`
2. **Enhanced Classes**: Extend existing TextNormalizer
3. **Configuration Files**: `config/advanced_normalization_config.yaml`
4. **Test Suites**: `tests/test_advanced_number_normalization.py`

### **Deployment Considerations**
1. **MCP Server Access**: External service connectivity requirements
2. **Fallback Strategy**: Graceful degradation if MCP unavailable
3. **Performance Monitoring**: Enhanced metrics for MCP processing
4. **Staging Environment**: MCP integration testing pipeline

---

## **🔄 Migration Strategy**

### **Phase 1: Research & Prototyping** (Week 1)
- Investigate available MCP NLP servers
- Create proof-of-concept integration
- Benchmark performance against current system
- Validate fallback mechanisms

### **Phase 2: Implementation** (Weeks 2-4)  
- Develop AdvancedTextNormalizer class
- Integrate with existing pipeline
- Comprehensive test suite development
- Configuration and deployment preparation

### **Phase 3: Production Rollout** (Month 2)
- Staged deployment with feature flags
- A/B testing against current system
- Performance monitoring and optimization
- Full production deployment

---

## **💡 Alternative Approaches Considered**

### **Option A: Enhanced Rule-Based (Fallback)**
```python
IDIOMATIC_PATTERNS = [
    r'\bone by one\b', r'\btwo by two\b', r'\bstep by step\b',
    r'\bone on one\b', r'\bday by day\b'
]
# Pro: No external dependencies, Con: Limited scalability
```

### **Option B: Hybrid MCP + Rules**
```python  
def process_with_hybrid_approach(self, text):
    # Try MCP first, fall back to enhanced rules
    try:
        return self.mcp_processor.analyze(text)
    except MCPError:
        return self.enhanced_rule_processor.analyze(text)
```

### **Option C: Full ML Pipeline (Future)**
- Train domain-specific transformer model
- Requires substantial data collection and training
- Higher complexity but maximum customization

---

**Technical Analysis Prepared**: August 11, 2025  
**Reviewed By**: Quinn (Senior Developer & QA Architect)  
**Status**: Ready for BMAD Team Architecture Review  
**Recommendation**: Proceed with Hybrid MCP + Rules approach (Option B)