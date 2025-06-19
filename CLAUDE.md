# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Groundit is a Python library that adds verifiability and trustworthiness to AI outputs by providing source references and confidence scores. It transforms AI extractions from black boxes into auditable, verifiable outputs.

## Commands

### Development Setup
```bash
# Install dependencies (uses uv)
uv sync

# Install with integration test dependencies
uv sync --group integration
```

### Testing
```bash
# Run all tests
pytest

# Run only unit tests (fast, no API calls)
pytest tests/unit/

# Run only integration tests (requires API keys)
pytest tests/integration/

# Run with specific markers
pytest -m "not integration"  # Skip integration tests
pytest -m "integration"      # Only integration tests
pytest -m "slow"            # Only slow tests

# Run single test file
pytest tests/unit/confidence/test_confidence_extractor.py
```

### Development Workflow
```bash
# No specific build, lint, or typecheck commands defined in project config
# Tests use pytest with modern importlib mode
```

## Architecture

### Core Design
Groundit follows a dual-module architecture:

1. **Reference Module** (`groundit.reference/`): Adds source tracking to extracted data
2. **Confidence Module** (`groundit.confidence/`): Quantifies model certainty using token probabilities

### Key Patterns

#### Type-Preserving Transformation
The reference module transforms Pydantic models to add source tracking while preserving original types:
- `create_model_with_source()` - Runtime model transformation
- `create_json_schema_with_source()` - Schema-level transformation for production
- `FieldWithSource[T]` - Generic container preserving type information

#### Syntax Tree to Confidence Mapping
The confidence module uses Lark parser to map JSON tokens to confidence scores:
- Parses JSON structure into syntax tree
- Maps each character to originating token
- Supports multiple aggregation strategies (sum, average, joint probability)

#### Integration Pipeline
Modules work together in a typical flow:
1. Transform schema with `create_model_with_source()`
2. Extract structured data with LLM using transformed schema
3. Add confidence scores with `add_confidence_scores()`
4. Add source spans with `add_source_spans()`

### Data Models

#### Reference Models
- `FieldWithSource[T]` - Core container with `value: T` and `source_quote: str`

#### Confidence Models
- `TokensWithLogprob` - Token with probability information
- `TopLogprob` - Alternative tokens with probabilities
- Uses OpenAI API response structures

### Important Design Decisions

1. **Non-Destructive Enrichment**: Both modules create enriched copies rather than modifying originals
2. **Dual API Strategy**: Runtime vs schema transformation for different use cases  
3. **Pluggable Aggregation**: `AggregationFunction` type enables different confidence calculation strategies
4. **Generic Type Integration**: Heavy use of Python typing for type safety through transformations

## Test Organization

- `tests/unit/` - Fast tests, no external dependencies
- `tests/integration/` - Tests requiring API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
- Tests mirror source code structure (`tests/unit/confidence/` matches `src/groundit/confidence/`)
- Integration tests marked with `@pytest.mark.integration` and `@pytest.mark.slow`
- Shared utilities in `tests/test_utils.py`

## API Key Setup
Integration tests require environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key" 
```

## Module Exports
Key functions available at package level:
- `get_confidence_scores()`, `add_confidence_scores()` - Confidence scoring
- `average_probability_aggregator()` - Aggregation function
- `add_source_spans()` - Source span tracking  
- `create_model_with_source()` - Model transformation