# Advanced Number Normalization Enhancement Plan

## Executive Summary
Current primitive number normalization incorrectly processes idiomatic expressions ("one by one" → "1 by 1"), requiring advanced linguistic intelligence through MCP integration.

## Problem Statement

### Current System Limitations
- **Over-aggressive conversion**: Treats all number words as mathematical quantities
- **Context blindness**: Cannot distinguish idiomatic expressions from numeric references  
- **Rule-based approach**: Simple find-and-replace without semantic understanding
- **Academic impact**: Affects readability and professionalism of spiritual content

### Specific Example
```
❌ Current: "And 1 by 1, he killed 6 of their children."
✅ Should be: "And one by one, he killed six of their children."
```

## MCP Integration Strategy

### Recommended MCP Servers

**1. `mcp-server-nlp` - Natural Language Processing**
- **Capability**: Context-aware linguistic analysis
- **Implementation**: Distinguish idiomatic vs mathematical contexts
- **Usage**: `mcp__nlp_analyze_text(text, preserve_idioms=True)`

**2. `mcp-server-spacy` - SpaCy NLP Integration**  
- **Capability**: Advanced tokenization with POS tagging
- **Implementation**: Identify grammatical roles of number words
- **Usage**: `mcp__spacy_process(text, model="en_core_web_sm")`

**3. `mcp-server-transformers` - Hugging Face Models**
- **Capability**: BERT/RoBERTa semantic understanding
- **Implementation**: Deep context analysis for number usage
- **Usage**: `mcp__transform_text(text, task="number-normalization")`

### Enhanced Architecture Design

```python
# Enhanced text_normalizer.py with MCP integration
class AdvancedTextNormalizer:
    def __init__(self):
        self.mcp_nlp = MCPClient("nlp-server")
        self.mcp_spacy = MCPClient("spacy-server")
        self.context_analyzer = ContextualNumberProcessor()
    
    def normalize_numbers_with_context(self, text):
        # Step 1: Linguistic analysis via MCP
        tokens = self.mcp_spacy.analyze_tokens(text)
        
        # Step 2: Context classification
        for token in tokens:
            if token.is_number_word:
                context = self.mcp_nlp.get_context_type(token)
                if context == "idiomatic":
                    continue  # Preserve "one by one"
                elif context == "mathematical":
                    text = self.convert_to_digit(token)
                elif context == "narrative":
                    text = self.apply_narrative_rules(token)
        
        return text
    
    def get_context_classification(self, number_phrase, surrounding_text):
        """Classify numerical context using MCP intelligence"""
        contexts = {
            'idiomatic': [r'\bone by one\b', r'\btwo by two\b', r'\bstep by step\b'],
            'mathematical': [r'\btwo thousand \d+\b', r'\bchapter \w+ verse \w+\b'],
            'temporal': [r'\byear \w+ thousand \w+\b', r'\bin \w+ \w+\b'],
            'narrative': [r'\bone of the\b', r'\btwo or three\b'],
            'scriptural': [r'\bverse \w+\b', r'\bchapter \w+\b']
        }
        
        # Use MCP for advanced classification
        return self.mcp_nlp.classify_context(number_phrase, contexts)
```

## Production Implementation Timeline

### IMMEDIATE (Week 1)
**Priority**: Critical Fix
- [ ] **Research MCP Ecosystem**: Investigate available NLP servers
- [ ] **Test MCP Integration**: Prototype with `mcp-server-spacy`
- [ ] **Immediate Fix**: Manual correction of "1 by 1" → "one by one" 
- [ ] **Fallback Strategy**: Ensure system works if MCP unavailable

**Deliverables**:
- MCP server compatibility assessment
- Prototype integration code
- Emergency fix for current content

### SHORT-TERM (Month 1)
**Priority**: System Enhancement  
- [ ] **Context-Aware Processing**: Full MCP integration implementation
- [ ] **Idiomatic Preservation**: Comprehensive expression database
- [ ] **Test Suite Enhancement**: Edge cases and linguistic scenarios
- [ ] **Performance Benchmarking**: MCP vs current system metrics

**Deliverables**:
- Enhanced `AdvancedTextNormalizer` class
- Comprehensive test coverage for number contexts
- Performance analysis and optimization

### LONG-TERM (Quarter 1)  
**Priority**: Domain Specialization
- [ ] **Sanskrit/Hindi Context Models**: Domain-specific training
- [ ] **Learning System**: Improve from user corrections
- [ ] **Academic Citation Handling**: Verse references, scholarly formats
- [ ] **Integration with Epic 4**: Scalability and production deployment

**Deliverables**:
- Domain-specific MCP models
- Adaptive learning pipeline
- Production-ready enhanced system

## Technical Specifications

### Context Classification Schema
```yaml
number_contexts:
  idiomatic:
    patterns: ["one by one", "two by two", "step by step"]
    action: "preserve"
    confidence_threshold: 0.9
  
  mathematical:
    patterns: ["two thousand five", "three hundred"]
    action: "convert_to_digit"
    confidence_threshold: 0.8
  
  scriptural:
    patterns: ["chapter X verse Y", "verse number"]  
    action: "partial_convert"
    confidence_threshold: 0.95
  
  narrative:
    patterns: ["one of the", "two or three"]
    action: "context_dependent"
    confidence_threshold: 0.7
```

### Integration Points
1. **Current System**: `src/utils/text_normalizer.py:convert_numbers()`
2. **MCP Interface**: New `src/utils/mcp_number_processor.py`
3. **Configuration**: `config/advanced_normalization_config.yaml`
4. **Testing**: `tests/test_advanced_number_normalization.py`

## Fallback Strategy

**If MCP Unavailable**:
```python
# Enhanced rule-based approach
CONTEXT_PATTERNS = {
    'idiomatic': [
        r'\bone by one\b', r'\btwo by two\b', r'\bstep by step\b',
        r'\bone on one\b', r'\bday by day\b', r'\bone at a time\b'
    ],
    'preserve_narrative': [
        r'\bone of the\b', r'\ball \w+ of them\b'
    ],
    'convert_mathematical': [
        r'\btwo thousand \d+\b', r'\bchapter \w+ verse \w+\b'
    ]
}
```

## Risk Assessment

### Technical Risks
- **MCP Dependency**: External service availability
- **Performance Impact**: Additional processing overhead  
- **Integration Complexity**: Multiple system coordination

### Mitigation Strategies
- **Graceful Degradation**: Fall back to enhanced rule-based system
- **Caching Strategy**: Cache MCP results for repeated phrases
- **Progressive Enhancement**: Implement in stages with rollback capability

## Success Metrics

### Quality Improvements
- **Idiomatic Preservation**: 100% correct handling of common expressions
- **Context Accuracy**: >95% correct classification of number usage
- **User Satisfaction**: Improved readability scores

### Technical Metrics  
- **Processing Time**: <2x current performance (maintain <2s target)
- **Memory Usage**: <50% increase from current baseline
- **Error Rate**: <1% false classifications

## Budget Considerations

### Development Time
- **Research & Prototyping**: 1 week
- **Implementation**: 2-3 weeks  
- **Testing & Integration**: 1 week
- **Documentation & Training**: 1 week

### Infrastructure  
- **MCP Server Costs**: Evaluate pricing models
- **Development Environment**: Enhanced with MCP testing capabilities
- **Production Deployment**: Staging environment for MCP integration testing

---

**Document Prepared**: August 11, 2025  
**Prepared By**: Quinn (Senior Developer & QA Architect)  
**Status**: Ready for Team Discussion  
**Next Action**: BMAD Team Review and Planning Session