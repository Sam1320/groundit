"""Integration tests for LLM calls with groundit confidence scoring.

These tests make actual API calls to LLM providers and verify that the
confidence scoring works with real model responses.
"""

import json
from rich.pretty import pprint
import pytest
from pydantic import BaseModel
from typing import Literal

from groundit.confidence.confidence_extractor import get_confidence_scores
from groundit.confidence.logprobs_aggregators import average_probability_aggregator
from tests.utils import create_confidence_model

class DiceRoll(BaseModel):
    """Result of a single dice roll with explicit confidence."""
    dice_result: int
    level_of_confidence: Literal["very high", "high", "medium", "low"]


class DiceExperiment(BaseModel):
    """Results of rolling three different dice with increasing uncertainty."""
    octahedron_roll: DiceRoll       # 8-sided die - medium confidence expected  
    dodecahedron_roll: DiceRoll     # 12-sided die - lowest confidence expected
    tetrahedron_roll: DiceRoll      # 4-sided die - highest confidence expected

DICE_EXPERIMENT_PROMPT = """Roll three dice for me:
1. A 4-sided die (tetrahedron) - faces numbered 1-4
2. An 8-sided die (octahedron) - faces numbered 1-8
3. A 12-sided die (dodecahedron) - faces numbered 1-12

For each roll, tell me the result and rate your confidence as 'very high', 'high', 'medium', or 'low'."""

class RiddleInfo(BaseModel):
    """Information extracted from a family riddle."""
    name: str
    gender: str
    number_of_brothers: int
    number_of_sisters: int


class RiddleAnalysis(BaseModel):
    """Analysis of a family riddle with confidence assessment."""
    extracted_info: RiddleInfo
    reasoning_confidence: Literal["very high", "high", "medium", "low"]

RIDDLE_EXPERIMENT_PROMPT = """Here's a family riddle: 'A girl named Marie has as many brothers as sisters,
but each brother has only half as many brothers as sisters.'
Extract Marie's name, gender, and determine how many brothers and sisters she has.
Also rate your confidence in your reasoning as 'very high', 'high', 'medium', or 'low'."""


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
class TestLLMConfidenceIntegration:
    """Integration tests for LLM confidence scoring with nested models."""

    @pytest.mark.parametrize("model", ["gpt-4.1-mini"])
    def test_dice_confidence_correlation(self, openai_client, model):
        """Test that confidence scores correlate with fundamental uncertainty.
        
        Rolling dice with different numbers of faces should show:
        - 4-sided die (tetrahedron): highest confidence
        - 8-sided die (octahedron): medium confidence  
        - 12-sided die (dodecahedron): lowest confidence
        """
        
        response = openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": DICE_EXPERIMENT_PROMPT
                }
            ],
            logprobs=True,
            response_format=DiceExperiment
        )
        
        dice_results = response.choices[0].message.parsed
        assert dice_results is not None
        
        # Validate dice roll ranges
        assert 1 <= dice_results.tetrahedron_roll.dice_result <= 4
        assert 1 <= dice_results.octahedron_roll.dice_result <= 8
        assert 1 <= dice_results.dodecahedron_roll.dice_result <= 12
        
        # Apply confidence scoring using probability aggregator for intuitive scores
        tokens = response.choices[0].logprobs.content
        content = response.choices[0].message.content
        
        confidence_scores = get_confidence_scores(
            json_string_tokens=tokens,
            aggregator=average_probability_aggregator  # Use probability space for intuitive scores
        )
        
        print("\n" + "="*50)
        print("DICE EXPERIMENT RESULTS")
        print("="*50)
        print("Original response:")
        pprint(json.loads(content), expand_all=True)
        print("\nConfidence scores (probabilities):")
        pprint(confidence_scores, expand_all=True)
        
        # Extract confidence scores for dice results
        tetrahedron_confidence = confidence_scores["tetrahedron_roll"]["dice_result"]
        octahedron_confidence = confidence_scores["octahedron_roll"]["dice_result"]
        dodecahedron_confidence = confidence_scores["dodecahedron_roll"]["dice_result"]
        
        print("\nConfidence Analysis:")
        print(f"Tetrahedron (4-sided):  {tetrahedron_confidence:.4f}")
        print(f"Octahedron (8-sided):   {octahedron_confidence:.4f}")
        print(f"Dodecahedron (12-sided): {dodecahedron_confidence:.4f}")
        
        # Check that confidence correlates with uncertainty (higher faces = lower confidence)
        # Note: This is a statistical trend, not a guarantee for every single run
        print("\nExpected trend: Tetrahedron > Octahedron > Dodecahedron")
        print(f"Actual trend: {tetrahedron_confidence:.4f} > {octahedron_confidence:.4f} > {dodecahedron_confidence:.4f}")
        
        # Validate structure using confidence model
        DiceExperimentConfidence = create_confidence_model(DiceExperiment)
        DiceExperimentConfidence.model_validate(confidence_scores)
        
        # The confidence scores should be valid probabilities (0-1 range)
        assert 0 <= tetrahedron_confidence <= 1
        assert 0 <= octahedron_confidence <= 1
        assert 0 <= dodecahedron_confidence <= 1

    @pytest.mark.parametrize("model", ["gpt-4.1-mini"])
    def test_riddle_extraction_confidence(self, openai_client, model):
        """Test confidence scores for data extraction vs logical reasoning.
        
        Should show high confidence for easily extracted info (name, gender)
        and lower confidence for logically derived info (sibling counts).
        """
        
        response = openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": RIDDLE_EXPERIMENT_PROMPT
                }
            ],
            logprobs=True,
            response_format=RiddleAnalysis
        )
        
        riddle_result = response.choices[0].message.parsed
        assert riddle_result is not None
        
        # Validate extracted information
        assert "marie" in riddle_result.extracted_info.name.lower()
        assert "girl" in riddle_result.extracted_info.gender.lower() or "female" in riddle_result.extracted_info.gender.lower()
        
        # Apply confidence scoring using probability aggregator
        tokens = response.choices[0].logprobs.content
        content = response.choices[0].message.content
        
        confidence_scores = get_confidence_scores(
            json_string_tokens=tokens,
            aggregator=average_probability_aggregator
        )
        
        print("\n" + "="*50)
        print("RIDDLE EXTRACTION EXPERIMENT")
        print("="*50)
        print("Original response:")
        pprint(json.loads(content), expand_all=True)
        print("\nConfidence scores (probabilities):")
        pprint(confidence_scores, expand_all=True)
        
        # Extract confidence scores for different types of information
        name_confidence = confidence_scores["extracted_info"]["name"]
        gender_confidence = confidence_scores["extracted_info"]["gender"]
        brothers_confidence = confidence_scores["extracted_info"]["number_of_brothers"]
        sisters_confidence = confidence_scores["extracted_info"]["number_of_sisters"]
        
        print("\nConfidence Analysis:")
        print(f"Name (direct extraction):     {name_confidence:.4f}")
        print(f"Gender (direct extraction):   {gender_confidence:.4f}")
        print(f"Brothers (logical reasoning): {brothers_confidence:.4f}")
        print(f"Sisters (logical reasoning):  {sisters_confidence:.4f}")
        
        # Calculate average confidence for extraction vs reasoning
        extraction_avg = (name_confidence + gender_confidence) / 2
        reasoning_avg = (brothers_confidence + sisters_confidence) / 2
        
        print("\nComparison:")
        print(f"Extraction average: {extraction_avg:.4f}")
        print(f"Reasoning average:  {reasoning_avg:.4f}")
        print("Expected: Extraction > Reasoning")
        print(f"Actual: {extraction_avg:.4f} {'>' if extraction_avg > reasoning_avg else '<='} {reasoning_avg:.4f}")
        
        # Validate structure using confidence model
        RiddleAnalysisConfidence = create_confidence_model(RiddleAnalysis)
        RiddleAnalysisConfidence.model_validate(confidence_scores)
        
        # All confidence scores should be valid probabilities
        assert 0 <= name_confidence <= 1
        assert 0 <= gender_confidence <= 1
        assert 0 <= brothers_confidence <= 1
        assert 0 <= sisters_confidence <= 1
    