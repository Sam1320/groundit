"""Pytest configuration and fixtures for integration tests."""

import os
import pytest
from typing import Generator


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """Get OpenAI API key from environment or skip test."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


@pytest.fixture(scope="session") 
def anthropic_api_key() -> str:
    """Get Anthropic API key from environment or skip test."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    return api_key


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