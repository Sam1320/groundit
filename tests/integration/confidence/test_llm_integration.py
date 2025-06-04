"""Integration tests for LLM calls with groundit confidence scoring.

These tests make actual API calls to LLM providers and verify that the
confidence scoring works with real model responses.
"""

import json
from rich.pretty import pprint
import pytest
from pydantic import BaseModel

from groundit.confidence.models import TokensWithLogprob, ChoiceLogprobs, Choice
from groundit.confidence.json_parser import replace_leaves_with_confidence_scores
from groundit.confidence.get_confidence_scores import map_characters_to_token_indices
from groundit.confidence.logprobs_aggregators import default_sum_aggregator
from tests.test_utils import create_confidence_model


class PersonExtraction(BaseModel):
    """Example model for testing structured extraction."""
    name: str
    age: int
    city: str


class CompanyInfo(BaseModel):
    """Example nested model for testing."""
    name: str
    founded_year: int
    industry: str
    employees: int | None = None


class Address(BaseModel):
    """Address information."""
    street: str
    city: str
    country: str


class ContactInfo(BaseModel):
    """Contact information with nested address."""
    email: str
    phone: str | None = None
    address: Address


@pytest.fixture
def openai_client(openai_api_key):
    """Create OpenAI client with API key."""
    try:
        import openai
    except ImportError:
        pytest.skip("OpenAI package not installed")
    
    return openai.OpenAI(api_key=openai_api_key)


@pytest.mark.integration
@pytest.mark.slow
class TestOpenAIIntegration:
    """Integration tests for OpenAI models."""

    @pytest.mark.parametrize("model", ["gpt-4.1"])
    def test_extract_person_with_confidence(self, openai_client, model):
        """Test extracting person info with confidence scores from OpenAI."""
        
        response = openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "Extract person information from the given text."
                },
                {
                    "role": "user",
                    "content": "John Smith is a 35-year-old software engineer living in San Francisco."
                }
            ],
            logprobs=True,
            response_format=PersonExtraction,
        )
        person = response.choices[0].message.parsed
        assert person is not None
        
        # Verify extraction worked using the structured response
        assert "john" in person.name.lower() or "smith" in person.name.lower()
        assert person.age == 35
        assert "francisco" in person.city.lower() or "san francisco" in person.city.lower()

        tokens = response.choices[0].logprobs.content
        content = response.choices[0].message.content
        
        # Apply confidence scoring
        confidence_scores = replace_leaves_with_confidence_scores(
            json_string=content,
            tokens=tokens,
            token_indices=map_characters_to_token_indices(tokens),
            aggregator=default_sum_aggregator
        )
        
        # Use confidence model for validation instead of manual checks
        PersonExtractionConfidence = create_confidence_model(PersonExtraction)
        PersonExtractionConfidence.model_validate(confidence_scores)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    @pytest.mark.parametrize("model", ["gpt-4.1"])
    def test_complete_confidence_workflow(self, openai_client, model):
        """Test the complete workflow from LLM call to confidence scores."""
        
        response = openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract company information from the given text."
                },
                {
                    "role": "user",
                    "content": "Tesla was founded by Elon Musk in 2003 and operates in the automotive industry."
                }
            ],
            logprobs=True,
            response_format=CompanyInfo,
        )
        
        company = response.choices[0].message.parsed
        assert company is not None
        assert "tesla" in company.name.lower()
        assert company.founded_year == 2003
        assert "automotive" in company.industry.lower()
        
        # Use tokens directly from OpenAI response
        tokens = response.choices[0].logprobs.content
        content = response.choices[0].message.content
        
        confidence_scores = replace_leaves_with_confidence_scores(
            json_string=content,
            tokens=tokens,
            token_indices=map_characters_to_token_indices(tokens),
            aggregator=default_sum_aggregator
        )
        
        # Use confidence model for proper validation
        CompanyInfoConfidence = create_confidence_model(CompanyInfo)
        CompanyInfoConfidence.model_validate(confidence_scores)
        
    
    @pytest.mark.parametrize("model", ["gpt-4.1"])
    def test_nested_model_confidence_scoring(self, openai_client, model):
        """Test confidence scoring with nested models."""
        
        response = openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract contact information from the given text."
                },
                {
                    "role": "user",
                    "content": "Contact John Doe at john.doe@email.com, phone 555-123-4567. He lives at 123 Main St, New York, USA."
                }
            ],
            logprobs=True,
            response_format=ContactInfo,
        )
        
        contact = response.choices[0].message.parsed
        assert contact is not None
        assert "@" in contact.email
        assert "usa" in contact.address.country.lower() or "united states" in contact.address.country.lower()
        
        # Apply confidence scoring to nested model
        tokens = response.choices[0].logprobs.content
        content = response.choices[0].message.content
        
        confidence_scores = replace_leaves_with_confidence_scores(
            json_string=content,
            tokens=tokens,
            token_indices=map_characters_to_token_indices(tokens),
            aggregator=default_sum_aggregator
        )
        pprint(json.loads(content), expand_all=True)
        pprint(confidence_scores, expand_all=True)
        # Use confidence model for nested structure validation
        ContactInfoConfidence = create_confidence_model(ContactInfo)
        ContactInfoConfidence.model_validate(confidence_scores)
    