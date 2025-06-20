"""Default configuration values for groundit."""

from groundit.confidence.logprobs_aggregators import average_probability_aggregator, AggregationFunction


DEFAULT_EXTRACTION_PROMPT = """Extract data from the following document based on the JSON schema.
Return null if the document does not contain information relevant to schema.
If the information is present implicitly, fill the source field with the text that contains the information.
Return only the JSON with no explanation text."""

DEFAULT_LLM_MODEL = "gpt-4.1"

DEFAULT_PROBABILITY_AGGREGATOR: AggregationFunction = average_probability_aggregator