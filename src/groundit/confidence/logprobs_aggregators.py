import math
from typing import Callable, TypeAlias


AggregationFunction: TypeAlias = Callable[[list[float]], float]


def default_sum_aggregator(logprobs: list[float]) -> float:
    """Default aggregation function that sums log probabilities (returns log probability)."""
    return sum(logprobs)


def average_probability_aggregator(logprobs: list[float]) -> float:
    """Aggregation function that computes the average probability (converts to probability space)."""
    if not logprobs:
        return 0.0
    probabilities = [math.exp(logprob) for logprob in logprobs]
    return round(sum(probabilities) / len(probabilities), 3)


def joint_probability_aggregator(logprobs: list[float]) -> float:
    """
    Aggregation function that computes the joint probability (converts to probability space).
    
    For independent events, joint probability is the product of individual probabilities.
    Since we have log probabilities, we sum them and then exponentiate.
    """
    return math.exp(sum(logprobs))