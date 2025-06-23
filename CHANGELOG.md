# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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