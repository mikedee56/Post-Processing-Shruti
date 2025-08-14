# Code Style and Conventions

## Language Standards
- **Python**: 3.10+ compatible syntax
- **Type Hints**: Present but not strictly enforced (some files use mypy: ignore-errors)
- **Docstrings**: Google-style docstrings used consistently
- **Imports**: Standard Python import organization

## Naming Conventions
- **Classes**: PascalCase (e.g., `SanskritPostProcessor`, `LexiconManager`)
- **Functions/Methods**: snake_case (e.g., `process_srt_file`, `identify_words`)
- **Variables**: snake_case (e.g., `config_path`, `fuzzy_threshold`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `FUZZY_MIN_CONFIDENCE`)
- **Private methods**: Leading underscore (e.g., `_load_config`, `_apply_corrections`)

## File Organization
- **Modules**: One main class per file with related utilities
- **Tests**: Mirror source structure with `test_` prefix
- **Configuration**: YAML files with descriptive names
- **Data Models**: Dataclasses with type hints

## Code Patterns
- **Error Handling**: Comprehensive try/catch with structured logging
- **Configuration**: YAML-based with environment overrides
- **Logging**: Structured logging with contextual information
- **Data Classes**: Used for structured data with type safety
- **Factory Pattern**: Used for component initialization
- **Dependency Injection**: Configuration-driven component setup

## Testing Patterns
- **pytest**: Primary testing framework
- **Test Structure**: Class-based tests with descriptive method names
- **Fixtures**: Extensive use of pytest fixtures for setup
- **Mocking**: Mock external dependencies
- **Coverage**: Aim for comprehensive test coverage

## Documentation Standards
- **Docstrings**: Required for all public methods and classes
- **Comments**: Explain complex logic and business rules
- **README**: YAML configuration documentation
- **Type Annotations**: Used consistently for method signatures

## Performance Considerations
- **Processing Targets**: <2 seconds per SRT file
- **Memory Management**: Efficient handling of large transcript files
- **Caching**: Used for lexicon and model data
- **Batch Processing**: Supported for large-scale operations