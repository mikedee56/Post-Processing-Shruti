# 7. Technical Architecture & Assumptions

### 7.1 Technology Stack
- **Runtime**: Python 3.10+
- **Data Processing**: pandas, NumPy
- **NLP Libraries**: iNLTK, IndicNLP Library for Indic language support
- **Specialized Models**: ByT5-Sanskrit (optional for advanced corrections)
- **Data Storage**: File-based approach with JSON/YAML lexicons
- **Version Control**: Git with comprehensive branching strategy

### 7.2 Repository Structure
**Architecture Type**: Monorepo

**Rationale**: Since this project involves a headless post-processing system with a potential future UI, a **Monorepo** structure with separate packages for the core logic, UI, and shared components provides:
- Easy code sharing and streamlined dependency management
- Unified versioning and deployment strategies
- Simplified cross-component integration testing

### 7.3 Service Architecture
**Architecture Type**: MVP Monolith with Progressive Complexity

The core of the MVP will be a **Monolith** service housing:cont
- Post-processing logic and algorithms
- Lexicon management systems
- Hybrid language identification
- QA metrics and reporting

**Evolution Path**: Designed for future refactoring into microservice or serverless architecture as scale requirements grow.

### 7.4 Testing Strategy
**Approach**: Full Testing Pyramid

**Components**:
- **Unit Tests**: Individual function validation
- **Integration Tests**: Complete pipeline testing
- **Golden Dataset Validation**: Accuracy measurements using manually perfected transcripts
- **Performance Tests**: Large-scale processing validation

### 7.5 Development Environment Assumptions
- **IDE Compatibility**: Optimized for Claude Code IDE capabilities
- **Configuration Management**: Single, version-controlled configuration file as source of truth
- **Fallback Handling**: Robust error handling for multi-word segments
- **Timestamp Integrity**: Absolute preservation of original SRT timing data

---

