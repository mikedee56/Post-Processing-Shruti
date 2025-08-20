# Testing & Quality Assurance Framework Guide

## Overview

This guide provides comprehensive documentation for the Story 5.5 Testing & Quality Assurance Framework, designed to support Epic 4 development and maintain high-quality standards across the ASR Post-Processing Workflow system.

## Framework Architecture

### Core Components

```
tests/
├── framework/               # Core testing infrastructure
│   ├── test_runner.py      # Enhanced test execution framework
│   └── __init__.py
├── unit/                   # Unit testing for individual modules
│   ├── test_all_modules.py # Comprehensive unit testing
│   └── __init__.py
├── integration/            # Integration and workflow testing
│   ├── test_end_to_end.py  # Complete workflow validation
│   └── __init__.py
├── performance/            # Performance and load testing
│   ├── test_performance_regression.py
│   └── __init__.py
├── data/                   # Test data management
│   ├── test_data_manager.py
│   ├── golden_dataset_validator.py
│   ├── test_fixtures.py
│   └── api_error_investigator.py
├── conftest.py            # Pytest configuration and fixtures
└── test_testing_framework_validation.py  # Framework validation

qa/
├── utils/                  # Quality validation utilities
│   └── quality_validator.py
├── metrics/               # Quality metrics collection
│   ├── quality_collector.py
│   └── quality_metrics_collector.py
├── dashboard/             # Quality monitoring and reporting
│   └── quality_dashboard.py
└── tools/                 # Quality assurance automation
    └── quality_checker.py
```

## Acceptance Criteria Implementation

### AC1: Comprehensive Test Coverage Implementation ✅

**Target**: 90%+ code coverage across all modules

**Implementation**:
- **Unit Tests**: Individual module testing with pytest
- **Integration Tests**: Component interaction validation
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Regression testing maintaining 10+ segments/sec

**Usage**:
```bash
# Run all tests with coverage
pytest --cov=src tests/

# Run specific test types
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/performance/   # Performance tests only

# Generate coverage report
pytest --cov=src --cov-report=html tests/
```

**Coverage Targets**:
- `post_processors/`: 95%+ coverage
- `utils/`: 93%+ coverage  
- `sanskrit_hindi_identifier/`: 91%+ coverage
- `ner_module/`: 88%+ coverage
- `qa/`: 94%+ coverage
- **Overall**: 92%+ coverage

### AC2: Quality Assurance Automation ✅

**Automated Tools**:
- **Linting**: flake8, black, isort for code style
- **Type Checking**: mypy for static type analysis
- **Security**: bandit for security vulnerability scanning
- **Dependencies**: safety for dependency vulnerability checking

**Quality Gates**:
```python
quality_gates = [
    QualityGate("coverage_gate", QualityCategory.COVERAGE, ">=", 90.0),
    QualityGate("performance_gate", QualityCategory.PERFORMANCE, ">=", 10.0),
    QualityGate("quality_gate", QualityCategory.CODE_QUALITY, ">=", 85.0)
]
```

**Usage**:
```bash
# Run quality checks
python qa/tools/quality_checker.py

# Individual tools
flake8 src/                    # Linting
mypy src/                      # Type checking  
bandit -r src/                 # Security scanning
safety check                   # Dependency vulnerabilities
```

### AC3: Test Data Management and Fixtures ✅

**Components**:
- **Golden Dataset**: Expert-verified content for accuracy validation
- **Synthetic Data**: Generated test data for edge cases
- **Test Fixtures**: Reusable test components and mocks
- **Data Management**: Automated test data lifecycle

**Test Data Types**:

1. **Golden Dataset**:
   ```python
   # Manually verified Sanskrit/Hindi content
   golden_data = {
       "accurate_transcripts": "Expert-validated SRT files",
       "sanskrit_terms": "Properly transliterated Sanskrit",
       "scripture_references": "Canonical verse text"
   }
   ```

2. **Synthetic Data**:
   ```python
   # Generated edge cases
   test_data_manager.generate_synthetic_srt_data(
       num_segments=100,
       include_edge_cases=True,
       sanskrit_density=0.3
   )
   ```

3. **Test Fixtures**:
   ```python
   @pytest.fixture
   def sanskrit_post_processor():
       """Configured SanskritPostProcessor for testing."""
       return SanskritPostProcessor(test_config)

   @pytest.fixture  
   def test_srt_file():
       """Sample SRT file with known content."""
       return create_test_srt_file()
   ```

### AC4: Continuous Integration Testing ✅

**CI/CD Integration**:
- **GitHub Actions**: Automated testing on code changes
- **Pre-commit Hooks**: Quality checks before code submission
- **Quality Gates**: Prevent deployment of failing code
- **Automated Deployment**: Testing in staging environments

**Pipeline Configuration**:
```yaml
# .github/workflows/quality-assurance.yml
name: Quality Assurance Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov flake8 mypy bandit safety
      - name: Run tests with coverage
        run: pytest --cov=src tests/
      - name: Run quality checks
        run: |
          flake8 src/
          mypy src/
          bandit -r src/
          safety check
```

**Pre-commit Hooks**:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
      - id: flake8
        name: flake8
        entry: flake8
        language: system
        types: [python]
```

### AC5: Quality Monitoring and Reporting ✅

**Real-time Monitoring**:
- **Quality Metrics**: Automated collection and analysis
- **Quality Dashboard**: Web-based real-time monitoring
- **Trend Analysis**: Historical quality tracking
- **Automated Alerts**: Quality degradation notifications

**Quality Dashboard**:
Access the quality dashboard at `http://localhost:8080` after starting:
```bash
python qa/dashboard/quality_dashboard.py
```

**Metrics Collection**:
```python
# Record quality metrics
metrics_collector.record_metric("code_coverage", 92.8, {"module": "post_processors"})
metrics_collector.record_metric("performance", 12.5, {"metric": "segments_per_second"})
metrics_collector.record_metric("quality_score", 89.3, {"category": "overall"})
```

**Alert Configuration**:
```python
# Set up quality alerts
alerts = [
    QualityAlert("coverage_low", "coverage", "<", 90.0),
    QualityAlert("performance_regression", "performance", "<", 10.0),
    QualityAlert("quality_degradation", "quality_score", "<", 85.0)
]
```

## Testing Strategies

### Unit Testing Strategy

**Scope**: Individual functions and classes
**Target Coverage**: 95%+
**Key Areas**:
- Text normalization functions
- Sanskrit/Hindi identification logic
- NER entity extraction
- Quality validation utilities

**Example**:
```python
def test_sanskrit_post_processor_initialization():
    """Test SanskritPostProcessor initializes correctly."""
    processor = SanskritPostProcessor()
    assert processor.enable_ner is True
    assert processor.ner_model is not None
    assert processor.capitalization_engine is not None

def test_text_normalization_scriptural_references():
    """Test conversion of scriptural references."""
    normalizer = AdvancedTextNormalizer()
    result = normalizer.convert_numbers_with_context("chapter two verse twenty five")
    assert result == "Chapter 2 verse 25"
```

### Integration Testing Strategy

**Scope**: Component interactions and workflows
**Target Coverage**: End-to-end processing paths
**Key Areas**:
- SRT file processing pipeline
- Sanskrit/Hindi correction workflow
- Quality assurance automation
- Performance monitoring integration

**Example**:
```python
def test_end_to_end_srt_processing():
    """Test complete SRT processing workflow."""
    processor = SanskritPostProcessor()
    
    # Process test SRT file
    metrics = processor.process_srt_file(test_input, test_output)
    
    # Validate results
    assert metrics.total_segments > 0
    assert metrics.segments_modified >= 0
    assert metrics.processing_time < 60.0  # Performance requirement
    
    # Validate output quality
    with open(test_output, 'r') as f:
        content = f.read()
        assert "Chapter 2 verse 25" in content  # Text normalization
        assert "Krishna" in content  # Capitalization
```

### Performance Testing Strategy

**Scope**: Performance regression detection
**Target**: 10+ segments/sec processing throughput
**Key Metrics**:
- Processing throughput (segments/sec)
- Memory usage during processing
- Response time for API calls
- Resource utilization monitoring

**Example**:
```python
def test_performance_regression():
    """Test processing performance meets requirements."""
    tester = PerformanceRegressionTester()
    
    # Run performance test
    results = tester.run_performance_test(
        test_file="data/test_samples/performance_test.srt",
        target_throughput=10.0
    )
    
    # Validate performance
    assert results['throughput'] >= 10.0
    assert results['memory_usage'] < 500  # MB
    assert results['variance'] <= 10.0  # Consistency requirement
```

### Golden Dataset Testing Strategy

**Scope**: Accuracy validation with expert content
**Target**: 95%+ accuracy against manual verification
**Key Areas**:
- Sanskrit term accuracy
- IAST transliteration correctness
- Scripture verse identification
- Overall processing quality

**Example**:
```python
def test_golden_dataset_accuracy():
    """Test processing accuracy against golden dataset."""
    validator = GoldenDatasetValidator()
    
    # Validate against expert-verified content
    results = validator.validate_accuracy(
        golden_dataset_path="data/golden_dataset/",
        processor=SanskritPostProcessor()
    )
    
    # Check accuracy targets
    assert results['overall_accuracy'] >= 95.0
    assert results['sanskrit_accuracy'] >= 98.0
    assert results['verse_accuracy'] >= 90.0
```

## Quality Standards

### Code Quality Standards

**Linting Requirements**:
- **flake8**: Maximum line length 88 characters
- **black**: Automatic code formatting
- **isort**: Import organization and sorting
- **mypy**: Type hint coverage 85%+

**Quality Metrics**:
- **Cyclomatic Complexity**: ≤ 10 per function
- **Code Coverage**: ≥ 90% overall
- **Documentation Coverage**: ≥ 80% for public APIs
- **Security Score**: No high-severity vulnerabilities

### Performance Standards

**Processing Performance**:
- **Throughput**: ≥ 10 segments/sec
- **Memory Usage**: ≤ 500MB for standard processing
- **Response Time**: ≤ 2 seconds for single segment
- **Variance**: ≤ 10% processing time consistency

**Quality Monitoring**:
- **Real-time Metrics**: Updated every 30 seconds
- **Alert Response**: ≤ 5 minutes for critical issues
- **Dashboard Availability**: 99.9% uptime
- **Trend Analysis**: 30-day rolling windows

### Security Standards

**Security Requirements**:
- **Vulnerability Scanning**: Zero high-severity issues
- **Dependency Security**: All dependencies current and secure
- **Code Security**: No hardcoded secrets or credentials
- **Input Validation**: All external inputs sanitized

## Testing Execution

### Local Development Testing

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Install testing dependencies
pip install pytest pytest-cov pytest-benchmark pytest-mock

# Run complete test suite
pytest --cov=src tests/

# Run specific test categories
pytest tests/unit/                    # Unit tests
pytest tests/integration/             # Integration tests  
pytest tests/performance/             # Performance tests
pytest tests/test_testing_framework_validation.py  # Framework validation

# Run quality assurance
python qa/tools/quality_checker.py

# Start quality dashboard
python qa/dashboard/quality_dashboard.py
```

### Continuous Integration Testing

**Automated Execution**:
- **On Push**: Full test suite execution
- **On Pull Request**: Quality gate validation
- **Nightly**: Extended performance testing
- **Weekly**: Golden dataset validation

**Quality Gates**:
1. **Code Coverage**: ≥ 90%
2. **Test Pass Rate**: ≥ 95%
3. **Performance**: ≥ 10 segments/sec
4. **Quality Score**: ≥ 85%
5. **Security**: Zero high-severity issues

### Production Validation

**Deployment Testing**:
- **Smoke Tests**: Basic functionality verification
- **Load Testing**: Performance under production load
- **Integration Testing**: External system compatibility
- **Rollback Testing**: Deployment rollback procedures

## Troubleshooting

### Common Issues

**Test Failures**:
1. **Import Errors**: Ensure PYTHONPATH includes `src/` directory
2. **Missing Dependencies**: Install with `pip install -r requirements.txt`
3. **Permission Issues**: Check file permissions for test data
4. **Resource Issues**: Ensure sufficient memory/disk space

**Performance Issues**:
1. **Slow Tests**: Use `pytest-benchmark` for performance profiling
2. **Memory Leaks**: Monitor with `memory_profiler` package
3. **Resource Contention**: Run tests in isolation
4. **Timeout Issues**: Increase timeout values for slow operations

**Quality Issues**:
1. **Coverage Gaps**: Use `coverage html` for detailed coverage reports
2. **Quality Regression**: Check quality dashboard trends
3. **Alert Fatigue**: Tune alert thresholds appropriately
4. **Dashboard Issues**: Restart quality dashboard service

### Debug Commands

```bash
# Run tests with verbose output
pytest -v tests/

# Run tests with debugging
pytest --pdb tests/

# Run specific test with output
pytest -s tests/test_specific_test.py::test_function

# Generate detailed coverage report
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html

# Profile test performance
pytest --benchmark-only tests/performance/

# Debug quality issues
python qa/tools/quality_checker.py --debug

# Check quality metrics
python qa/metrics/quality_metrics_collector.py --report
```

## Epic 4 Readiness

### Framework Capabilities

The testing framework provides robust foundation for Epic 4 development:

**Scalability**:
- **Parallel Testing**: Support for distributed test execution
- **Load Testing**: High-volume processing validation
- **Performance Monitoring**: Real-time performance tracking
- **Quality Assurance**: Automated quality validation

**Reliability**:
- **Comprehensive Coverage**: 90%+ test coverage
- **Regression Detection**: Automated regression testing
- **Quality Gates**: Prevent quality degradation
- **Monitoring**: Real-time system health monitoring

**Maintainability**:
- **Documentation**: Comprehensive testing documentation
- **Training Materials**: Developer onboarding guides
- **Best Practices**: Established testing patterns
- **Quality Standards**: Clear quality expectations

### Epic 4 Integration Points

**MCP Pipeline Excellence Integration**:
- **API Testing**: MCP service integration testing
- **Performance Testing**: Pipeline performance validation
- **Quality Monitoring**: End-to-end quality tracking
- **Deployment Testing**: MCP deployment validation

## Training and Documentation

### Developer Onboarding

**Required Reading**:
1. This Testing Framework Guide
2. `docs/architecture.md` - System architecture
3. `qa/README.md` - Quality assurance processes
4. Story 5.5 documentation

**Hands-on Training**:
1. Run complete test suite locally
2. Write unit tests for new features
3. Use quality dashboard for monitoring
4. Practice CI/CD workflows

### Best Practices

**Test Writing**:
- Write tests before implementing features (TDD)
- Use descriptive test names and documentation
- Keep tests isolated and independent
- Mock external dependencies appropriately

**Quality Assurance**:
- Run quality checks before committing code
- Monitor quality dashboard regularly
- Address quality alerts promptly
- Maintain quality standards consistently

**Performance**:
- Profile performance regularly
- Monitor resource usage
- Optimize bottlenecks proactively
- Maintain performance benchmarks

## Conclusion

The Story 5.5 Testing & Quality Assurance Framework provides a comprehensive foundation for maintaining high-quality standards while supporting Epic 4 development. The framework ensures:

- **90%+ Code Coverage** across all modules
- **Automated Quality Assurance** with real-time monitoring
- **Comprehensive Test Data Management** with golden dataset validation
- **CI/CD Integration** with automated quality gates
- **Real-time Quality Monitoring** with trend analysis and alerting

This framework is ready to support the complex development requirements of Epic 4: MCP Pipeline Excellence while maintaining the high quality standards required for production deployment.

---

*This guide is part of the Story 5.5 implementation and will be updated as the framework evolves.*