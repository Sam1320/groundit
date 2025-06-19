"""Pytest configuration and fixtures for integration tests."""

import os
import pytest


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """Get OpenAI API key from environment or skip test."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def openai_client(openai_api_key):
    """Create OpenAI client with API key."""
    try:
        import openai
    except ImportError:
        pytest.skip("OpenAI package not installed")
    
    return openai.OpenAI(api_key=openai_api_key)


@pytest.fixture(scope="session")
def test_document():
    """Load the test document from the data directory."""
    test_doc_path = os.path.join(os.path.dirname(__file__), "..", "data", "example_doc.txt")
    with open(test_doc_path, "r", encoding="utf-8") as f:
        return f.read()


# Pytest markers for integration tests
pytestmark = [
    pytest.mark.integration,  # Mark all tests in this directory as integration
    pytest.mark.slow,         # Mark them as slow (requires network calls)
]


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requires API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (makes network calls)"
    ) 