from typing import Any, TypeAlias
import json
from dataclasses import dataclass

from lark import Lark, Token, Transformer_NonRecursive, Tree, v_args
from lark.tree import Meta
from groundit.confidence.models import TokensWithLogprob
from groundit.confidence.logprobs_aggregators import (
    AggregationFunction,
    default_sum_aggregator,
)

PyTree: TypeAlias = (
    Any  # a tree-like structure built out of container-like Python objects.
)


# Define a grammar for JSON
json_grammar = r"""
    start: value

    ?value: object              #'?' is a Lark convention indicating that the rule can return the value directly instead of creating a separate parse tree node.
          | array
          | string
          | SIGNED_NUMBER -> number    #'-> number' specifies an alias for the rule
          | "true"        -> true
          | "false"       -> false
          | "null"        -> null

    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : key ":" value
    key    : ESCAPED_STRING

    string : ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""


def _map_characters_to_token_indices(
    extracted_data_token: list[TokensWithLogprob],
) -> list[int]:
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
        >>> _map_characters_to_token_indices(tokens)
        [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]
    """

    token_indices = []

    for token_idx, token_data in enumerate(extracted_data_token):
        token_text = token_data.token
        token_indices.extend([token_idx] * len(token_text))

    return token_indices


# Transformer that processes the tree and substitutes each atomic value with the cumulative log-probability of its tokens
@v_args(meta=True)
class Extractor(Transformer_NonRecursive):
    def __init__(
        self,
        tokens: list[TokensWithLogprob],
        token_indices: list[int],
        aggregator: AggregationFunction = default_sum_aggregator,
        debug: bool = False,
    ):
        super().__init__()
        self.tokens = tokens
        self.token_indices = token_indices
        self.aggregator = aggregator
        self.debug = debug

    def _extract_token_logprobs(self, start_pos: int, end_pos: int) -> list[float]:
        """Extract log probabilities for tokens corresponding to character positions."""
        token_start = self.token_indices[start_pos]
        token_end = self.token_indices[end_pos]
        if self.debug:
            print(
                "tokens being aggregated",
                [self.tokens[i].token for i in range(token_start, token_end)],
            )
        return [self.tokens[i].logprob for i in range(token_start, token_end)]

    def _compute_aggregated_value(self, start_pos: int, end_pos: int) -> float:
        """Compute aggregated value using the configured aggregation function."""
        logprobs = self._extract_token_logprobs(start_pos, end_pos)
        return self.aggregator(logprobs)

    def number(self, meta: Meta, children: list[Token]) -> float:
        return self._compute_aggregated_value(meta.start_pos, meta.end_pos)

    def string(self, meta: Meta, children: list[Token]) -> float:
        # Adjust positions to exclude opening and closing quotes
        content_start = meta.start_pos + 1  # Skip opening quote
        content_end = meta.end_pos - 1  # Skip closing quote
        return self._compute_aggregated_value(content_start, content_end)

    def true(self, meta: Meta, children: list[Token]) -> float:
        return self._compute_aggregated_value(meta.start_pos, meta.end_pos)

    def false(self, meta: Meta, children: list[Token]) -> float:
        return self._compute_aggregated_value(meta.start_pos, meta.end_pos)

    def null(self, meta: Meta, children: list[Token]) -> float:
        return self._compute_aggregated_value(meta.start_pos, meta.end_pos)

    def array(self, meta: Meta, children: list[Any]) -> list[float]:
        return children

    def object(self, meta: Meta, children: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in children:
            result[key] = value
        return result

    def pair(self, meta: Meta, children: list[Any]) -> tuple[str, Any]:
        value = children[1]
        key = children[0]
        if (
            isinstance(value, Tree) and not value.children
        ):  # ['b', Tree(Token('RULE', 'value'), [])]
            value = None
        return key, value

    def key(self, meta: Meta, children: list[Token]) -> str:
        return children[0][1:-1]

    def start(self, meta: Meta, children: list[dict[str, Any]]) -> dict[str, Any]:
        return children[0]


def _replace_leaves_with_confidence_scores(
    json_string: str,
    tokens: list[TokensWithLogprob],
    token_indices: list[int],
    aggregator: AggregationFunction = default_sum_aggregator,
    debug: bool = False,
) -> PyTree:
    """
    Extracts JSON data from a JSON string using a Lark parser.

    Args:
        json_string (str): The JSON string to parse.
        tokens (list[ChatCompletionTokenLogprob]): The tokens to use for log probability extraction.
        token_indices (list[int]): A list of integers where each position corresponds to a character in the concatenated JSON string,
        and the integer at each position is the index of the token responsible for generating that specific character.
        aggregator (AggregationFunction): The function to use for aggregating log probabilities.

    Returns:
        PyTree: The parsed JSON data.
    """

    json_parser = Lark(
        json_grammar, parser="lalr", propagate_positions=True, maybe_placeholders=False
    )
    tree = json_parser.parse(json_string)
    if debug:
        print(tree.pretty())
    extractor = Extractor(tokens, token_indices, aggregator, debug)
    return extractor.transform(tree)


# ---------------------------------------------------------------------------
# Tokenizer-specific configuration


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    """Configuration for cleaning/normalisation steps per tokenizer family."""

    # Mapping from sentinel char -> replacement byte (1 char) used *in order*
    sentinel_replacements: dict[str, str]
    # Tokens that signal end-of-sequence for that tokenizer/model family
    stop_tokens: list[str]


GPT2_SPEC = TokenizerSpec(
    sentinel_replacements={
        "Ċ": "\n",  # newline sentinel
        "Ġ": " ",  # space sentinel
    },
    stop_tokens=["</s>"],
)


SENTENCE_PIECE_SPEC = TokenizerSpec(
    sentinel_replacements={
        "▁": " ",  # space sentinel
    },
    stop_tokens=["<end_of_turn>"],
)


BYTE_BPE_SPEC = TokenizerSpec(
    sentinel_replacements={},
    stop_tokens=[],  # OpenAI logprobs don't include explicit EOS
)


def _identify_tokenizer_spec(model_name: str | None) -> TokenizerSpec:
    """Return the :class:`TokenizerSpec` appropriate for *model_name*.

    The heuristic is simple string matching; this can be extended later if we
    encounter more edge-cases.
    """

    if model_name is None:
        # Fallback to BYTE_BPE_SPEC (default for openai models)
        return BYTE_BPE_SPEC

    lower = model_name.lower()

    # OpenAI models: "openai/…" or anything containing "gpt-4" / "gpt-3" etc.
    if lower.startswith("openai/") or "gpt-" in lower:
        return BYTE_BPE_SPEC

    # SentencePiece-family (Gemma, Llama, etc.)
    if any(x in lower for x in ("gemma", "llama", "deepseek")):
        return SENTENCE_PIECE_SPEC

    # Default for remaining HF instruction models (Mistral, Mixtral, GPT-NeoX …)
    return GPT2_SPEC


def _clean_tokens(
    tokens: list[TokensWithLogprob],
    spec: TokenizerSpec,
) -> list[TokensWithLogprob]:
    # Fast-path: nothing to strip
    if not spec.stop_tokens:
        return tokens

    clean: list[TokensWithLogprob] = []
    for t in tokens:
        if t.token in spec.stop_tokens:
            break  # stop at first EOS-like token
        clean.append(t)
    return clean


# type: ignore[override]
def _normalise_raw_json(raw: str, spec: TokenizerSpec) -> str:
    text = raw
    for sentinel, repl in spec.sentinel_replacements.items():
        text = text.replace(sentinel, repl)
    return text


def get_confidence_scores(
    json_string_tokens: list[TokensWithLogprob],
    model_name: str | None = None,
    aggregator: AggregationFunction = default_sum_aggregator,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Takes a list of tokens representing a JSON string and returns the same JSON string with the leaves replaced with confidence scores.

    Args:
        json_string_tokens: A list of TokensWithLogprob objects representing a JSON string
        model_name: The name of the model used to extract the JSON string (needed to identify the tokenizer family)
        aggregator: The function to use for aggregating log probabilities
        debug: Whether to print debug information

    Returns:
        The parsed JSON data with leaves replaced by confidence scores
    """

    # ------------------------------------------------------------------
    # 1. Drop EOS token(s) and obtain a clean token list
    # ------------------------------------------------------------------

    spec = _identify_tokenizer_spec(model_name)

    clean_tokens = _clean_tokens(json_string_tokens, spec)

    # ------------------------------------------------------------------
    # 2. Join tokens and normalise whitespace sentinels so that the string
    #    becomes valid JSON that the downstream Lark grammar can parse.
    # ------------------------------------------------------------------

    raw_json_string = "".join(t.token for t in clean_tokens)

    json_string = _normalise_raw_json(raw_json_string, spec)

    # Validate that we ended up with syntactically correct JSON – this keeps
    # error semantics identical to the previous implementation.
    _validate_json_string_tokens_internal(json_string)

    # ------------------------------------------------------------------
    # 3. Character-to-token mapping and confidence extraction
    # ------------------------------------------------------------------

    token_indices = _map_characters_to_token_indices(clean_tokens)

    return _replace_leaves_with_confidence_scores(
        json_string=json_string,
        tokens=clean_tokens,
        token_indices=token_indices,
        aggregator=aggregator,
        debug=debug,
    )


def add_confidence_scores(
    extraction_result: dict[str, Any],
    tokens: list[TokensWithLogprob],
    *,
    model_name: str | None = None,
    aggregator: AggregationFunction = default_sum_aggregator,
    debug: bool = False,
) -> dict[str, Any]:
    """
    adds a "confidence" field at the same level of existing leaf fields with a _confidence suffix
    e.g.
    input:
    {
        "names": [
            {
                "value": "John Doe",
                "source_quote": "the name of the patient is John Doe"
            },
            {
                "value": "Jane Doe",
                "source_quote": "the name of the patient is Jane Doe"
            }
        ],
        "age": {
            "value": 30,
            "source_quote": "the age of the patient is 30"
        },
        "city": {
            "value": "New York",
            "source_quote": "the city of the patient is New York"
        }
    }
    output:
    {
        "names": [
            {
            "value": "John Doe",
            "source_quote": "the name of the patient is John Doe",
            "value_confidence": 0.90,
            "source_quote_confidence": 0.90
            },
            {
                "value": "Jane Doe",
                "source_quote": "the name of the patient is Jane Doe",
                "value_confidence": 0.90,
                "source_quote_confidence": 0.90
            }
        ],
        "age": {
            "value": 30,
            "source_quote": "the age of the patient is 30",
            "value_confidence": 0.90,
            "source_quote_confidence": 0.90
        },
        "city": {
            "value": "New York",
            "source_quote": "the city of the patient is New York",
            "value_confidence": 0.90,
            "source_quote_confidence": 0.90
        }
    }
    """
    import copy

    confidence_scores = get_confidence_scores(
        json_string_tokens=tokens,
        model_name=model_name,
        aggregator=aggregator,
        debug=debug,
    )

    def add_confidence_recursive(data, confidence_data):
        """Recursively add confidence scores to fields"""

        if isinstance(data, dict) and isinstance(confidence_data, dict):
            # Create a list of items to avoid modifying dict during iteration
            items_to_process = list(data.items())

            # Process each field in the dict
            for key, value in items_to_process:
                if key in confidence_data:
                    # If value is a nested structure, recurse into it
                    if isinstance(value, (dict, list)):
                        add_confidence_recursive(value, confidence_data[key])
                    else:
                        # Only add confidence score for leaf values (non-dict, non-list)
                        data[f"{key}_confidence"] = confidence_data[key]

        elif isinstance(data, list) and isinstance(confidence_data, list):
            # Handle lists
            for i, item in enumerate(data):
                if i < len(confidence_data):
                    add_confidence_recursive(item, confidence_data[i])

    # Create a deep copy to avoid modifying the original
    enriched_result = copy.deepcopy(extraction_result)

    # Add confidence scores to the copy
    add_confidence_recursive(enriched_result, confidence_scores)

    return enriched_result


def _validate_json_string_tokens_internal(json_string: str) -> None:
    """Raise :class:`ValueError` if *json_string* is not valid JSON."""

    try:
        json.loads(json_string)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The token list does not represent a valid JSON string"
        ) from exc
