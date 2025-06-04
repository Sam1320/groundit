import json
from collections import defaultdict
import pytest
import tiktoken
from pydantic import BaseModel
from groundit.confidence.logprobs_aggregators import default_sum_aggregator
from groundit.confidence.models import TokensWithLogprob
from groundit.confidence.json_parser import replace_leaves_with_confidence_scores
from groundit.confidence.get_confidence_scores import map_characters_to_token_indices
from rich.pretty import pprint


# Pydantic models for testing
class FlatModel(BaseModel):
    name: str | float
    age: str | float
    city: str | float


class Preferences(BaseModel):
    theme: str | float
    notifications: bool | float
    marketing_emails: bool | float

class Profile(BaseModel):
    name: str | float
    preferences: Preferences
    bio: str | None


class Stats(BaseModel):
    posts: int | str | float
    followers: int | str | float


class User(BaseModel):
    profile: Profile
    stats: Stats


class Metadata(BaseModel):
    created: str | float
    version: str | float


class NestedModel(BaseModel):
    user: User
    metadata: Metadata


TEST_OBJECTS = [
    # Flat model
    FlatModel(
        name="John",
        age=30,
        city="New York"
    ),
    # Nested model
    NestedModel(
        user=User(
            profile=Profile(
                name="Alice",
                preferences=Preferences(
                    theme="dark",
                    notifications=True,
                    marketing_emails=False
                ),
                bio=None
            ),
            stats=Stats(
                posts=42,
                followers=1337
            )
        ),
        metadata=Metadata(
            created="2024-01-01",
            version="1.0"
        )
    )
]


@pytest.fixture
def flat_json() -> str:
    return TEST_OBJECTS[0].model_dump_json()

@pytest.fixture
def nested_json() -> str:
    return TEST_OBJECTS[1].model_dump_json()


@pytest.fixture
def flat_dict() -> dict:
    return TEST_OBJECTS[0].model_dump()

@pytest.fixture
def nested_dict() -> dict:
    return TEST_OBJECTS[1].model_dump()


def string_to_tokens(text: str) -> list[TokensWithLogprob]:
    """
    Convert a string to a list of TokensWithLogprob using tiktoken.
    
    Args:
        text: The input string to tokenize
        
    Returns:
        List of TokensWithLogprob objects with logprob set to -1.0
    """
    # Use OpenAI's tokenizer (cl100k_base is used by GPT-3.5/GPT-4)
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Get token IDs
    token_ids = enc.encode(text)
    
    tokens = []
    for token_id in token_ids:
        tokens.append(
            TokensWithLogprob(
                token=enc.decode([token_id]),
                bytes=list(enc.decode_bytes([token_id])),
                logprob=-1.0,  # Set to -1 for now
                top_logprobs=None 
            )
        )
    
    return tokens

def test_create_tokens_from_string():
    json_string = '{"key1": "value1", "key2": "value2"}'
    tokens = string_to_tokens(json_string)
    assert json_string == "".join([token.token for token in tokens])
    assert len(tokens) > 1

def test_replace_leaves_with_confidence_scores_flat(flat_json):
    input_json = json.loads(flat_json)
    output_json = replace_leaves_with_confidence_scores(
        json_string=flat_json,
        tokens=string_to_tokens(flat_json),
        token_indices=map_characters_to_token_indices(string_to_tokens(flat_json)),
        aggregator=default_sum_aggregator
    )

    FlatModel.model_validate(input_json)
    FlatModel.model_validate(output_json)


def test_replace_leaves_with_confidence_scores_nested(nested_json):
    input_json = json.loads(nested_json)
    output_json = replace_leaves_with_confidence_scores(
        json_string=nested_json,
        tokens=string_to_tokens(nested_json),
        token_indices=map_characters_to_token_indices(string_to_tokens(nested_json)),
        aggregator=default_sum_aggregator
    )

    NestedModel.model_validate(input_json)
    NestedModel.model_validate(output_json)


def test_aggregator_with_manual_logprobs():
    """Test that the aggregator correctly sums logprobs for a multi-token string."""
    
    # Create the JSON string we want to test
    json_string = '{ "name" : "multitokenstring"}'
    
    # Create tokens manually - break down "multitokenstring" into parts with specific logprobs
    logprob_dict = defaultdict(lambda: -0.1, {'multi': -1.0, 'token': -2.0, 'string': -3.0})

    manual_tokens = [
        TokensWithLogprob(token='{ "', bytes=[], logprob=logprob_dict['{ "'], top_logprobs=None),
        TokensWithLogprob(token='name', bytes=[], logprob=logprob_dict['name'], top_logprobs=None),
        TokensWithLogprob(token='" : "', bytes=[], logprob=logprob_dict['" : "'], top_logprobs=None),
        TokensWithLogprob(token='multi', bytes=[], logprob=logprob_dict['multi'], top_logprobs=None),  # Part 1 of "multitokenstring"
        TokensWithLogprob(token='token', bytes=[], logprob=logprob_dict['token'], top_logprobs=None),  # Part 2 of "multitokenstring"  
        TokensWithLogprob(token='string', bytes=[], logprob=logprob_dict['string'], top_logprobs=None), # Part 3 of "multitokenstring"
        TokensWithLogprob(token='"}', bytes=[], logprob=logprob_dict['"}'], top_logprobs=None),
    ]
    
    # Verify our tokens reconstruct the original string
    reconstructed = "".join([token.token for token in manual_tokens])
    assert reconstructed == json_string, f"Expected: {json_string}, Got: {reconstructed}"
    
    # Apply the confidence score replacement
    output_json = replace_leaves_with_confidence_scores(
        json_string=json_string,
        tokens=manual_tokens,
        token_indices=map_characters_to_token_indices(manual_tokens),
        aggregator=default_sum_aggregator,
        debug=True
    )
    
    expected_confidence = default_sum_aggregator([logprob_dict["multi"], logprob_dict["token"], logprob_dict["string"]])
    confidence_score = output_json["name"]
    
    assert confidence_score == expected_confidence, (
        f"Confidence aggregation bug detected!\n"
        f"JSON: {json_string}\n"
        f"Tokens: {[(t.token, t.logprob) for t in manual_tokens]}\n"
        f"Expected confidence (content only): {expected_confidence}\n"
        f"Actual confidence (includes quotes): {confidence_score}\n"
        f"Output JSON: {output_json}"
    )

