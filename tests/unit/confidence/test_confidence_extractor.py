import json
from collections import defaultdict
import pytest
import tiktoken
from pydantic import BaseModel, ValidationError

from groundit.confidence.logprobs_aggregators import default_sum_aggregator
from groundit.confidence.models import TokensWithLogprob
from groundit.confidence.confidence_extractor import get_confidence_scores, add_confidence_scores
from tests.utils import create_confidence_model
from tests.models import (NestedModel, TEST_OBJECT)


@pytest.fixture
def nested_json() -> str:
    return TEST_OBJECT.model_dump_json()


def string_to_tokens(text: str) -> list[TokensWithLogprob]:
    """Convert a string to a list of TokensWithLogprob using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    
    tokens = []
    for token_id in token_ids:
        tokens.append(
            TokensWithLogprob(
                token=enc.decode([token_id]),
                bytes=list(enc.decode_bytes([token_id])),
                logprob=-1.0,
                top_logprobs=None 
            )
        )
    
    return tokens

def test_create_tokens_from_string():
    json_string = '{"key1": "value1", "key2": "value2"}'
    tokens = string_to_tokens(json_string)
    assert json_string == "".join([token.token for token in tokens])
    assert len(tokens) > 1

def test_replace_leaves_with_confidence_scores_nested(nested_json):
    input_json = json.loads(nested_json)
    output_json = get_confidence_scores(
        json_string_tokens=string_to_tokens(nested_json),
        aggregator=default_sum_aggregator
    )

    # Validate structure matches original model
    NestedModel.model_validate(input_json)
    
    # Create confidence model and validate confidence output
    NestedModelConfidence = create_confidence_model(NestedModel)
    NestedModelConfidence.model_validate(output_json)


def test_aggregator_with_manual_logprobs():
    """Test that the aggregator correctly sums logprobs for a multi-token string."""
    
    json_string = '{ "name" : "multitokenstring"}'
    logprob_dict = defaultdict(lambda: -0.1, {'multi': -1.0, 'token': -2.0, 'string': -3.0})

    manual_tokens = [
        TokensWithLogprob(token='{ "', bytes=[], logprob=logprob_dict['{ "'], top_logprobs=None),
        TokensWithLogprob(token='name', bytes=[], logprob=logprob_dict['name'], top_logprobs=None),
        TokensWithLogprob(token='" : "', bytes=[], logprob=logprob_dict['" : "'], top_logprobs=None),
        TokensWithLogprob(token='multi', bytes=[], logprob=logprob_dict['multi'], top_logprobs=None),
        TokensWithLogprob(token='token', bytes=[], logprob=logprob_dict['token'], top_logprobs=None),  
        TokensWithLogprob(token='string', bytes=[], logprob=logprob_dict['string'], top_logprobs=None),
        TokensWithLogprob(token='"}', bytes=[], logprob=logprob_dict['"}'], top_logprobs=None),
    ]
    
    # Verify tokens reconstruct the original string
    reconstructed = "".join([token.token for token in manual_tokens])
    assert reconstructed == json_string
    
    output_json = get_confidence_scores(
        json_string_tokens=manual_tokens,
        aggregator=default_sum_aggregator
    )
    
    expected_confidence = default_sum_aggregator([logprob_dict["multi"], logprob_dict["token"], logprob_dict["string"]])
    confidence_score = output_json["name"]
    
    assert confidence_score == expected_confidence


def test_confidence_model_validation():
    """Test that confidence models provide stricter validation than manually modified models."""
    
    # Sample confidence output with None (simulating old bug)
    invalid_confidence_output = {
        'user': {
            'profile': {
                'name': -1.5,
                'preferences': {
                    'theme': None,  # This should fail validation
                    'notifications': -2.0,
                    'marketing_emails': -1.0
                },
                'bio': -0.5
            },
            'stats': {
                'posts': -1.8,
                'followers': -2.2
            }
        },
        'metadata': {
            'created': -1.0,
            'version': -1.5
        }
    }
    
    # Manual model incorrectly allows None
    class ManuallyModifiedModel(BaseModel):
        user: dict
        metadata: dict
    
    # Auto-generated confidence model is stricter
    NestedModelConfidence = create_confidence_model(NestedModel)
    
    # Manual model incorrectly accepts None
    ManuallyModifiedModel.model_validate(invalid_confidence_output)
    
    # Auto-generated model correctly rejects None
    with pytest.raises(ValidationError):
        NestedModelConfidence.model_validate(invalid_confidence_output)


def test_add_confidence_scores_simple():
    """Test add_confidence_scores with simple nested dict structure."""
    
    # Sample extraction result with nested structure
    extraction_result = {
        "first_name": {
            "value": "John",
            "source_quote": "First name: John"
        },
        "last_name": {
            "value": "Doe", 
            "source_quote": "Last name: Doe"
        }
    }
    
    # Mock tokens for testing
    json_string = '{"first_name":{"value":"John","source_quote":"First name: John"},"last_name":{"value":"Doe","source_quote":"Last name: Doe"}}'
    tokens = string_to_tokens(json_string)
    
    result = add_confidence_scores(
        extraction_result=extraction_result,
        tokens=tokens,
        aggregator=default_sum_aggregator
    )
    
    # Check that confidence fields are added at leaf level only
    assert "value_confidence" in result["first_name"]
    assert "source_quote_confidence" in result["first_name"]
    assert "value_confidence" in result["last_name"]
    assert "source_quote_confidence" in result["last_name"]
    
    # Check that no confidence fields are added at root level
    assert "first_name_confidence" not in result
    assert "last_name_confidence" not in result
    
    # Check that confidence values are floats
    assert isinstance(result["first_name"]["value_confidence"], float)
    assert isinstance(result["first_name"]["source_quote_confidence"], float)


def test_add_confidence_scores_flat_structure():
    """Test add_confidence_scores with flat structure (leaf values only)."""
    
    extraction_result = {
        "name": "Alice",
        "age": 25,
        "city": "New York"
    }
    
    json_string = '{"name":"Alice","age":25,"city":"New York"}'
    tokens = string_to_tokens(json_string)
    
    result = add_confidence_scores(
        extraction_result=extraction_result,
        tokens=tokens,
        aggregator=default_sum_aggregator
    )
    
    # Check that confidence fields are added for leaf values
    assert "name_confidence" in result
    assert "age_confidence" in result
    assert "city_confidence" in result
    
    # Original values should remain
    assert result["name"] == "Alice"
    assert result["age"] == 25
    assert result["city"] == "New York"


def test_add_confidence_scores_with_lists():
    """Test add_confidence_scores with list structures."""
    
    extraction_result = {
        "items": [
            {"name": "item1", "value": 10},
            {"name": "item2", "value": 20}
        ]
    }
    
    json_string = '{"items":[{"name":"item1","value":10},{"name":"item2","value":20}]}'
    tokens = string_to_tokens(json_string)
    
    result = add_confidence_scores(
        extraction_result=extraction_result,
        tokens=tokens,
        aggregator=default_sum_aggregator
    )
    
    # Check that confidence fields are added to leaf values in list items
    assert "name_confidence" in result["items"][0]
    assert "value_confidence" in result["items"][0]
    assert "name_confidence" in result["items"][1]
    assert "value_confidence" in result["items"][1]
    
    # Check that no confidence field is added for the list itself
    assert "items_confidence" not in result


def test_add_confidence_scores_empty_dict():
    """Test add_confidence_scores with empty dict."""
    
    extraction_result = {}
    tokens = string_to_tokens('{}')
    
    result = add_confidence_scores(
        extraction_result=extraction_result,
        tokens=tokens,
        aggregator=default_sum_aggregator
    )
    
    assert result == {}

