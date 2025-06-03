from typing import Any, TypeAlias
from rich import print

from lark import Lark, Token, Transformer_NonRecursive, Tree, v_args
from lark.tree import Meta
from pydantic import BaseModel
from groundit.confidence.models import TokensWithLogprob
from groundit.confidence.logprobs_aggregators import AggregationFunction, default_sum_aggregator

PyTree: TypeAlias = Any  # a tree-like structure built out of container-like Python objects.


class HasProb(BaseModel):
    value: Any
    start: int
    end: int
    logprob: float


# Define a grammar for JSON
json_grammar = r"""
    start: value

    ?value: object              #'?' is a Lark convention indicating that the rule can return the value directly instead of creating a separate parse tree node.
          | array
          | string
          | SIGNED_NUMBER -> number    #'-> number' specifies an alias for the rule
          | "true"
          | "false"
          | "null"

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
            print("tokens being aggregated", [self.tokens[i].token for i in range(token_start, token_end)])
        return [self.tokens[i].logprob for i in range(token_start, token_end)]

    def _compute_aggregated_value(self, start_pos: int, end_pos: int) -> float:
        """Compute aggregated value using the configured aggregation function."""
        logprobs = self._extract_token_logprobs(start_pos, end_pos)
        return self.aggregator(logprobs)

    def number(self, meta: Meta, children: list[Token]) -> float:
        return self._compute_aggregated_value(meta.start_pos, meta.end_pos)

    def string(self, meta: Meta, children: list[Token]) -> float:
        return self._compute_aggregated_value(meta.start_pos, meta.end_pos)

    def true(self, meta: Meta, children: list[Token]) -> float:
        return self._compute_aggregated_value(meta.start_pos, meta.end_pos)

    def false(self, meta: Meta, children: list[Token]) -> float:
        return self._compute_aggregated_value(meta.start_pos, meta.end_pos)

    def null(self, meta: Meta, children: list[Token]) -> None:
        return None

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
        if isinstance(value, Tree) and not value.children:  # ['b', Tree(Token('RULE', 'value'), [])]
            value = None
        return key, value

    def key(self, meta: Meta, children: list[Token]) -> str:
        return children[0][1:-1]

    def start(self, meta: Meta, children: list[dict[str, Any]]) -> dict[str, Any]:
        return children[0]


def replace_leaves_with_confidence_scores(
    json_string: str, 
    tokens: list[TokensWithLogprob], 
    token_indices: list[int], 
    aggregator: AggregationFunction = default_sum_aggregator
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

    json_parser = Lark(json_grammar, parser="lalr", propagate_positions=True, maybe_placeholders=False)
    tree = json_parser.parse(json_string)
    extractor = Extractor(tokens, token_indices, aggregator)
    return extractor.transform(tree)
