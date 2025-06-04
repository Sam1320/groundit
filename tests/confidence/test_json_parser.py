import json
from collections import defaultdict
import pytest
import tiktoken
from pydantic import BaseModel, create_model
from groundit.confidence.logprobs_aggregators import default_sum_aggregator
from groundit.confidence.models import TokensWithLogprob
from groundit.confidence.json_parser import replace_leaves_with_confidence_scores
from groundit.confidence.get_confidence_scores import map_characters_to_token_indices
from pydantic import ValidationError


def create_confidence_model(original_model: type[BaseModel]) -> type[BaseModel]:
    """
    Creates a new Pydantic model where all leaf fields (non-nested BaseModel fields) 
    are transformed to accept float values for confidence scores.
    
    Args:
        original_model: The original Pydantic model to transform
        
    Returns:
        A new model class with the same structure but float types for leaf fields
    """
    def transform_field_type(field_info) -> tuple:
        """Transform a field's type annotation to include float for confidence scores."""
        field_type = field_info.annotation
        
        # Check if it's a BaseModel subclass (nested model) - keep as is
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return (create_confidence_model(field_type), field_info.default)
        
        # For leaf fields, allow float (confidence scores)
        return (float, field_info.default)
    
    # Transform all fields
    transformed_fields = {}
    for field_name, field_info in original_model.model_fields.items():
        transformed_fields[field_name] = transform_field_type(field_info)
    
    # Create new model with transformed fields
    confidence_model_name = f"{original_model.__name__}Confidence"
    return create_model(confidence_model_name, **transformed_fields)


# Pydantic models for testing
class FlatModel(BaseModel):
    name: str
    age: int  
    city: str


class Preferences(BaseModel):
    theme: str
    notifications: bool
    marketing_emails: bool

class Profile(BaseModel):
    name: str
    preferences: Preferences
    bio: str | None


class Stats(BaseModel):
    posts: int
    followers: int


class User(BaseModel):
    profile: Profile
    stats: Stats


class Metadata(BaseModel):
    created: str
    version: str


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

def test_replace_leaves_with_confidence_scores_flat(flat_json):
    input_json = json.loads(flat_json)
    output_json = replace_leaves_with_confidence_scores(
        json_string=flat_json,
        tokens=string_to_tokens(flat_json),
        token_indices=map_characters_to_token_indices(string_to_tokens(flat_json)),
        aggregator=default_sum_aggregator
    )

    # Validate structure matches original model
    FlatModel.model_validate(input_json)
    
    # Create confidence model and validate confidence output  
    FlatModelConfidence = create_confidence_model(FlatModel)
    FlatModelConfidence.model_validate(output_json)


def test_replace_leaves_with_confidence_scores_nested(nested_json):
    input_json = json.loads(nested_json)
    output_json = replace_leaves_with_confidence_scores(
        json_string=nested_json,
        tokens=string_to_tokens(nested_json),
        token_indices=map_characters_to_token_indices(string_to_tokens(nested_json)),
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
    
    output_json = replace_leaves_with_confidence_scores(
        json_string=json_string,
        tokens=manual_tokens,
        token_indices=map_characters_to_token_indices(manual_tokens),
        aggregator=default_sum_aggregator
    )
    
    expected_confidence = default_sum_aggregator([logprob_dict["multi"], logprob_dict["token"], logprob_dict["string"]])
    confidence_score = output_json["name"]
    
    assert confidence_score == expected_confidence


def test_confidence_model_validation():
    """Test that confidence models provide stricter validation than manually modified models."""
    
    # Sample confidence output with None (simulating old bug)
    invalid_confidence_output = {
        'name': -1.5,
        'age': None,  # This should fail validation
        'city': -0.8
    }
    
    # Manual model incorrectly allows None
    class ManuallyModifiedModel(BaseModel):
        name: str | float
        age: int | float | None  # Allows None - not what we want for confidence scores
        city: str | float
    
    # Auto-generated confidence model is stricter
    FlatModelConfidence = create_confidence_model(FlatModel)
    
    # Manual model incorrectly accepts None
    ManuallyModifiedModel.model_validate(invalid_confidence_output)
    
    # Auto-generated model correctly rejects None
    with pytest.raises(ValidationError):
        FlatModelConfidence.model_validate(invalid_confidence_output)

