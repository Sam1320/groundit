"""Integration tests for groundit.

These tests make actual calls to external services (LLMs) and should be run
separately from unit tests. They require API keys and network connectivity.

Run integration tests with:
    pytest tests/integration/

Skip integration tests:
    pytest tests/confidence/  # (unit tests only)
""" 