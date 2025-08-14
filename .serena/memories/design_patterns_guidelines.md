# Design Patterns and Guidelines

## Architectural Patterns

### MVP Monolith Approach
- Single repository with progressive complexity
- Story-driven development (Epic 1 → Epic 2 → Epic 3 → Epic 4)
- File-based data storage with JSON/YAML lexicons
- Configuration-driven component initialization

### Processing Pipeline Pattern
- **SanskritPostProcessor**: Main orchestrator
- **TextNormalizer/AdvancedTextNormalizer**: Text processing core
- **LexiconManager**: Dictionary and term management
- **MetricsCollector**: Statistics and quality tracking

### Context-Aware Processing
- **Context Classification**: IDIOMATIC, SCRIPTURAL, TEMPORAL, SANSKRIT, MATHEMATICAL
- **MCP Integration**: Model Context Protocol for enhanced intelligence
- **Fallback Strategies**: Graceful degradation when services unavailable
- **Circuit Breaker**: Reliability patterns for external dependencies

## Development Guidelines

### Story-Driven Development
- Each story has acceptance criteria and QA requirements
- Stories build progressively: 2.1 → 2.2 → 2.3 → 2.4.x
- Backward compatibility must be maintained
- Performance targets: <2 seconds processing time

### Configuration Management
- YAML-based configuration files in `config/` directory
- Environment variable overrides supported
- Default configurations work out-of-the-box
- Externalized lexicons for linguistic expert updates

### Quality Assurance
- Comprehensive test coverage with pytest
- Golden dataset for accuracy measurements (WER/CER reduction)
- Processing quality validation with semantic drift detection
- Metrics collection for continuous improvement

### Security and Compliance
- Defensive security - academic content processing only
- IAST transliteration standards compliance
- Academic integrity maintenance
- Version-controlled lexicons for traceability

### Performance Patterns
- Sub-second processing targets
- Caching for lexicon and model data
- Batch processing capabilities
- Memory-efficient handling of large files
- Asynchronous processing where beneficial

## Key Anti-Patterns to Avoid
- Don't break backward compatibility
- Don't compromise performance for features
- Don't hardcode linguistic rules (use externalized lexicons)
- Don't skip comprehensive testing
- Don't modify core architecture without story-based planning