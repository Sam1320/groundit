"""Default configuration values for groundit."""

from groundit.confidence.logprobs_aggregators import (
    average_probability_aggregator,
    AggregationFunction,
)


DEFAULT_EXTRACTION_PROMPT = """Extract data from the following document based on the JSON schema.
Return null *only if* the document clearly does *not* contain information relevant to the schema.
If the information is present implicitly, fill the source field with the text that contains the information.
Return only the JSON with no explanation text."""

DEFAULT_VERBALIZED_CONFIDENCE_PROMPT = """Extract data from the following document based on the JSON schema.
Return null *only if* the document clearly does *not* contain information relevant to the schema.
If the information is present implicitly, fill the source field with the text that contains the information.

For each extracted value and source quote, also provide your confidence as a decimal between 0 and 1:
- value_verbalized_confidence: How confident you are that the extracted value is correct (0.0 = no confidence, 1.0 = completely confident)
- source_quote_verbalized_confidence: How confident you are that the source quote accurately represents where the information came from (0.0 = no confidence, 1.0 = completely confident)

Return only the JSON with no explanation text."""

DEFAULT_LLM_MODEL = "gpt-4.1"

DEFAULT_PROBABILITY_AGGREGATOR: AggregationFunction = average_probability_aggregator
