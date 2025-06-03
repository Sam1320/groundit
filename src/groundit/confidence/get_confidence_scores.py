from typing import Any
import json

from groundit.confidence.models import TokensWithLogprob

from groundit.confidence.json_parser import replace_leaves_with_confidence_scores, AggregationFunction, default_sum_aggregator



def map_characters_to_token_indices(extracted_data_token: list[TokensWithLogprob]) -> list[int]:
    """
    Maps each character in the JSON string output to its corresponding token index.

    Args:
    extracted_data_token : A list of `TokenLogprob` objects, where each object represents a token and its associated data.

    Returns:
    A list of integers where each position corresponds to a character in the concatenated JSON string,
    and the integer at each position is the index of the token responsible for generating that specific character.
    Example:
        >>> tokens = [ChatCompletionTokenLogprob(token='{'),
                      ChatCompletionTokenLogprob(token='"key1"'),
                      ChatCompletionTokenLogprob(token=': '),
                      ChatCompletionTokenLogprob(token='"value1"'),
                      ChatCompletionTokenLogprob(token='}')]
        >>> map_characters_to_token_indices(tokens)
        [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]
    """

    token_indices = []

    for token_idx, token_data in enumerate(extracted_data_token):
        token_text = token_data.token
        token_indices.extend([token_idx] * len(token_text))

    return token_indices


def get_confidence_scores(
    json_string_tokens: list[TokensWithLogprob],
    aggregator: AggregationFunction = default_sum_aggregator
) -> dict[str, Any]:
    """Takes a list of tokens representing a JSON string and returns the same JSON string with the leaves replaced with confidence scores"""
    json_string = "".join([logprob.token for logprob in json_string_tokens])
    try:
        json.loads(json_string)
    except json.JSONDecodeError:
        raise ValueError("The token list does not represent a valid JSON string")
    token_indices = map_characters_to_token_indices(json_string_tokens)

    return replace_leaves_with_confidence_scores(
        json_string=json_string,
        tokens=json_string_tokens,
        token_indices=token_indices,
        aggregator=aggregator
    )
