"""Unit tests for auto-switching to verbalized confidence."""

import warnings
from unittest.mock import Mock, patch
from pydantic import BaseModel, Field

from groundit import groundit, _supports_logprobs


class ExampleModel(BaseModel):
    name: str = Field(description="Test name")


class TestSupportsLogprobs:
    """Test the _supports_logprobs function."""

    def test_gpt_models_support_logprobs(self):
        """Test that GPT models are detected as supporting logprobs."""
        assert _supports_logprobs("openai/gpt-4") is True
        assert _supports_logprobs("gpt-3.5-turbo") is True
        assert _supports_logprobs("GPT-4") is True

    def test_mistral_models_support_logprobs(self):
        """Test that Mistral models are detected as supporting logprobs."""
        assert _supports_logprobs("huggingface/mistral-7b") is True
        assert _supports_logprobs("mistral-small") is True
        assert _supports_logprobs("MISTRAL-LARGE") is True

    def test_claude_models_dont_support_logprobs(self):
        """Test that Claude models are detected as not supporting logprobs."""
        assert _supports_logprobs("anthropic/claude-3-sonnet") is False
        assert _supports_logprobs("claude-sonnet-4") is False
        assert _supports_logprobs("CLAUDE-HAIKU") is False

    def test_unknown_models_default_to_supporting_logprobs(self):
        """Test that unknown models default to supporting logprobs."""
        assert _supports_logprobs("unknown-model") is True
        assert _supports_logprobs("custom/model") is True


class TestAutoVerbalizationSwitching:
    """Test automatic switching to verbalized confidence for non-logprob models."""

    @patch("groundit.LiteLLM")
    def test_auto_switch_for_claude_model(self, mock_litellm):
        """Test that Claude models automatically switch to verbalized confidence."""
        # Mock the LiteLLM client and response
        mock_client = Mock()
        mock_litellm.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = '{"name": {"value": "test", "source_quote": "test", "value_confidence": 0.9, "source_quote_confidence": 0.9}}'
        mock_client.chat.completions.create.return_value = mock_response

        # Test with Claude model and verbalized_confidence=False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch("groundit.add_source_spans") as mock_add_source:
                mock_add_source.return_value = {
                    "name": {
                        "value": "test",
                        "source_quote": "test",
                        "value_confidence": 0.9,
                        "source_quote_confidence": 0.9,
                    }
                }

                groundit(
                    document="test document",
                    extraction_model=ExampleModel,
                    llm_model="anthropic/claude-3-sonnet",
                    verbalized_confidence=False,  # Should be auto-switched to True
                )

                # Check that warning was issued
                assert len(w) == 1
                assert "does not support logprobs" in str(w[0].message)
                assert "Automatically enabling verbalized confidence mode" in str(
                    w[0].message
                )

                # Check that the call was made without logprobs
                call_args = mock_client.chat.completions.create.call_args
                assert (
                    "logprobs" not in call_args[1]
                )  # Should not have logprobs in kwargs

    @patch("groundit.LiteLLM")
    def test_no_auto_switch_for_gpt_model(self, mock_litellm):
        """Test that GPT models don't automatically switch to verbalized confidence."""
        # Mock the LiteLLM client and response
        mock_client = Mock()
        mock_litellm.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = '{"name": {"value": "test", "source_quote": "test"}}'
        mock_response.choices[0].logprobs.content = []
        mock_client.chat.completions.create.return_value = mock_response

        # Test with GPT model and verbalized_confidence=False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with (
                patch("groundit.add_confidence_scores") as mock_add_confidence,
                patch("groundit.add_source_spans") as mock_add_source,
            ):
                mock_add_confidence.return_value = {
                    "name": {
                        "value": "test",
                        "source_quote": "test",
                        "value_confidence": 0.9,
                        "source_quote_confidence": 0.9,
                    }
                }
                mock_add_source.return_value = {
                    "name": {
                        "value": "test",
                        "source_quote": "test",
                        "value_confidence": 0.9,
                        "source_quote_confidence": 0.9,
                    }
                }

                groundit(
                    document="test document",
                    extraction_model=ExampleModel,
                    llm_model="openai/gpt-4",
                    verbalized_confidence=False,  # Should remain False
                )

                # Check that no warning was issued
                assert len(w) == 0

                # Check that the call was made with logprobs
                call_args = mock_client.chat.completions.create.call_args
                assert call_args[1]["logprobs"] is True

    @patch("groundit.LiteLLM")
    def test_explicit_verbalized_confidence_overrides_auto_switch(self, mock_litellm):
        """Test that explicitly setting verbalized_confidence=True works for any model."""
        # Mock the LiteLLM client and response
        mock_client = Mock()
        mock_litellm.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = '{"name": {"value": "test", "source_quote": "test", "value_confidence": 0.9, "source_quote_confidence": 0.9}}'
        mock_client.chat.completions.create.return_value = mock_response

        # Test with GPT model and explicit verbalized_confidence=True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch("groundit.add_source_spans") as mock_add_source:
                mock_add_source.return_value = {
                    "name": {
                        "value": "test",
                        "source_quote": "test",
                        "value_confidence": 0.9,
                        "source_quote_confidence": 0.9,
                    }
                }

                groundit(
                    document="test document",
                    extraction_model=ExampleModel,
                    llm_model="openai/gpt-4",
                    verbalized_confidence=True,  # Explicitly set to True
                )

                # Check that no warning was issued (no auto-switching needed)
                assert len(w) == 0

                # Check that the call was made without logprobs
                call_args = mock_client.chat.completions.create.call_args
                assert (
                    "logprobs" not in call_args[1]
                )  # Should not have logprobs in kwargs
