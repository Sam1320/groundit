# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6] - 2025-01-07

### Added
- **Automatic model detection**: `groundit()` now automatically detects models that don't support logprobs and enables verbalized confidence mode with a user warning

### Changed
- **Unified confidence field names**: Both logprob and verbalized confidence now use consistent field names (`value_confidence` and `source_quote_confidence`) for simplified frontend integration

## [0.1.5] - 2025-01-07

### Added
- **Verbalized Confidence Support**: Support for models that don't provide logprobs (like Claude/Anthropic models)
  - `FieldWithSourceAndConfidence` model with `value_verbalized_confidence` and `source_quote_verbalized_confidence` fields
  - `verbalized_confidence` parameter to `groundit()` function (defaults to `False` for backward compatibility)
  - `DEFAULT_VERBALIZED_CONFIDENCE_PROMPT` for requesting confidence scores directly from models
  - Integration tests for verbalized confidence path with Claude Sonnet

### Changed
- `create_model_with_source()` and `create_json_schema_with_source()` now accept `enrichment_class` parameter to support different field enrichment types
- `groundit()` function conditionally handles logprobs parameter to support models that don't accept it

## [0.1.4] - 2025-06-29

### Added
- Support for other models that provide `logprobs` (e.g. Mistral-Instruct, Gemma).  Confidence scoring now auto-detects tokenizer families and cleans sentinel characters accordingly.
- Unit test covering GPT-2 style sentinel handling.

### Changed
- `confidence_extractor` refactored: model-aware cleaning/normalisation via `TokenizerSpec`.
- `add_confidence_scores` and `groundit` now accept `model_name` so downstream logic can pick the right tokenizer behaviour.

## [0.1.2] - 2025-06-24

### Added
- Enhanced Literal type support in model transformations
- Improved JSON schema generation for Literal types

### Changed
- Refined extraction prompts for less false negatives when extracting data
- Updated project version to 0.1.2

## [0.1.0] - 2025-06-23

### Added
- Initial release of Groundit library
- `groundit()` function for unified data extraction pipeline with confidence scores and source tracking
- Reference module for adding source quotes to extracted data
- Confidence module for token-level probability analysis
- Support for both Pydantic models and JSON schemas
- Multiple probability aggregation strategies (average, joint, sum)
- Type-preserving transformations with `FieldWithSource[T]`
- Comprehensive test suite with unit and integration tests

### Features
- Source tracking: Links extracted values to original document text
- Confidence scoring: Token probability analysis for trustworthiness metrics
- Schema transformation: Runtime and compile-time model enhancement
- OpenAI API integration with structured outputs and logprobs
- Configurable extraction prompts and models
- Support for Python 3.12+

[0.1.0]: https://github.com/username/groundit/releases/tag/v0.1.0
