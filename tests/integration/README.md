# Integration Tests

This directory contains integration tests that make actual API calls to LLM providers to test groundit's confidence scoring with real model responses.

## Setup

### 1. Install Integration Dependencies

```bash
# Install the optional integration dependencies
uv sync --group integration
```

### 2. Set API Keys

Set the required environment variables for the LLM providers you want to test:

```bash
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"
```

## Running Tests

### Run All Integration Tests
```bash
pytest tests/integration/
```

### Run Only OpenAI Integration Tests
```bash
pytest tests/integration/ -k "OpenAI"
```

### Run Only Fast Unit Tests (Skip Integration)
```bash
pytest tests/unit/  # Only unit tests
# or
pytest -m "not integration"  # Skip integration tests
```

### Run with Verbose Output
```bash
pytest tests/integration/ -v -s
```

## Test Structure

The tests are now organized with clear separation:

- `tests/unit/confidence/` - Unit tests for `src/groundit/confidence/`
- `tests/integration/confidence/` - Integration tests for `src/groundit/confidence/`
- `tests/test_utils.py` - Shared test utilities (like `create_confidence_model`)

Integration tests are marked with `@pytest.mark.integration` and `@pytest.mark.slow`, automatically skip if API keys are missing, and verify both extraction accuracy and confidence score generation.

## Best Practices Followed

1. **Clear Separation**: Unit tests in `tests/unit/`, integration tests in `tests/integration/`
2. **Markers**: Tests are properly marked for filtering
3. **Environment**: Tests gracefully handle missing API keys
4. **Structure**: Tests mirror source code organization
5. **Naming**: Clear, descriptive test names following `test_*` convention
6. **Documentation**: Each test has clear docstrings explaining purpose
7. **Fixtures**: Reusable fixtures for common setup (API clients, etc.)
8. **Parametrization**: Tests multiple models/scenarios where appropriate
9. **Shared Utilities**: Common test utilities in `tests/test_utils.py`

## Adding New Integration Tests

When adding new integration tests:

1. Place them in the appropriate subdirectory under `tests/integration/`
2. Use descriptive names: `test_[what_is_being_tested]_[scenario].py`
3. Mark with `@pytest.mark.integration` and `@pytest.mark.slow`
4. Include proper docstrings
5. Handle missing dependencies/API keys gracefully
6. Test both the extraction and confidence scoring aspects
7. Use shared utilities from `tests.test_utils` for consistency 