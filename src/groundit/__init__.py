import json
import warnings
from typing import Any, Type, Optional
from pydantic import BaseModel
from litellm import LiteLLM

from groundit.confidence.confidence_extractor import add_confidence_scores
from groundit.confidence.logprobs_aggregators import (
    AggregationFunction,
    average_probability_aggregator,
    joint_probability_aggregator,
)
from groundit.reference.add_source_spans import add_source_spans
from groundit.reference.create_model_with_source import (
    create_model_with_source,
    create_json_schema_with_source,
)
from groundit.reference.models import FieldWithSource, FieldWithSourceAndConfidence
from groundit.config import (
    DEFAULT_EXTRACTION_PROMPT,
    DEFAULT_VERBALIZED_CONFIDENCE_PROMPT,
    DEFAULT_LLM_MODEL,
    DEFAULT_PROBABILITY_AGGREGATOR,
)


def _supports_logprobs(model_name: str) -> bool:
    """
    Determine if a model supports logprobs based on its name.

    Args:
        model_name: The model name/identifier

    Returns:
        True if the model supports logprobs, False otherwise
    """
    model_lower = model_name.lower()

    # Models that support logprobs
    logprob_models = ["gpt", "mistral"]

    # Models that don't support logprobs
    non_logprob_models = ["claude"]

    # Check for non-logprob models first
    for model in non_logprob_models:
        if model in model_lower:
            return False

    # Check for logprob-supporting models
    for model in logprob_models:
        if model in model_lower:
            return True

    # Default to True for unknown models
    return True


def _build_request_params(
    llm_model: str,
    extraction_prompt: str,
    document: str,
    response_format: Any,
    verbalized_confidence: bool,
) -> dict[str, Any]:
    """Build request parameters for LLM API call."""
    params = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": document},
        ],
        "response_format": response_format,
    }

    # Only add logprobs for non-verbalized confidence
    if not verbalized_confidence:
        params["logprobs"] = True

    return params


def _process_llm_response(
    response: Any,
    document: str,
    llm_model: str,
    probability_aggregator: AggregationFunction,
    verbalized_confidence: bool,
) -> dict[str, Any]:
    """Process LLM response and add confidence scores and source spans."""
    # Parse the response
    content = response.choices[0].message.content
    extraction_result = json.loads(content)

    if verbalized_confidence:
        # For verbalized confidence, skip logprob-based confidence scoring
        result_with_confidence = extraction_result
    else:
        # Add confidence scores from logprobs
        tokens = response.choices[0].logprobs.content
        result_with_confidence = add_confidence_scores(
            extraction_result=extraction_result,
            tokens=tokens,
            model_name=llm_model,
            aggregator=probability_aggregator,
        )

    # Add source spans
    final_result = add_source_spans(result_with_confidence, document)
    return final_result


def groundit(
    document: str,
    extraction_model: Optional[Type[BaseModel]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    extraction_prompt: Optional[str] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    probability_aggregator: AggregationFunction = DEFAULT_PROBABILITY_AGGREGATOR,
    litellm_client: Optional[LiteLLM] = None,
    verbalized_confidence: bool = False,
) -> dict[str, Any]:
    """
    Complete groundit pipeline for data extraction with confidence scores and source tracking.

    This function orchestrates the full groundit workflow:
    1. Transform schema to include source tracking
    2. Extract data using LLM with transformed schema
    3. Add confidence scores (either from logprobs or verbalized confidence)
    4. Add source spans linking extracted values to document text

    Args:
        document: The source document to extract information from
        extraction_model: Pydantic model class for structured extraction (takes precedence if both provided)
        extraction_schema: JSON schema dict for extraction (used if extraction_model not provided)
        extraction_prompt: System prompt for guiding the extraction (uses default if None)
        llm_model: model to use for extraction
        probability_aggregator: Function to aggregate token probabilities into confidence scores
        litellm_client: litellm client instance (creates default if None)
        verbalized_confidence: If True, use verbalized confidence instead of logprobs (default: False)

    Returns:
        Dictionary with extracted data enriched with confidence scores and source quotes

    Raises:
        ValueError: If neither extraction_model nor extraction_schema are provided
    """
    if litellm_client is None:
        litellm_client = LiteLLM()

    # Auto-enable verbalized confidence for models that don't support logprobs
    if not verbalized_confidence and not _supports_logprobs(llm_model):
        verbalized_confidence = True
        warnings.warn(
            f"Model '{llm_model}' does not support logprobs. "
            "Automatically enabling verbalized confidence mode.",
            UserWarning,
            stacklevel=2,
        )

    if extraction_prompt is None:
        extraction_prompt = (
            DEFAULT_VERBALIZED_CONFIDENCE_PROMPT
            if verbalized_confidence
            else DEFAULT_EXTRACTION_PROMPT
        )

    # Choose enrichment class based on confidence method
    enrichment_class = (
        FieldWithSourceAndConfidence if verbalized_confidence else FieldWithSource
    )

    if extraction_model is not None:
        # Use Pydantic model approach
        model_with_source = create_model_with_source(extraction_model, enrichment_class)

        request_params = _build_request_params(
            llm_model=llm_model,
            extraction_prompt=extraction_prompt,
            document=document,
            response_format=model_with_source,
            verbalized_confidence=verbalized_confidence,
        )

        response = litellm_client.chat.completions.create(**request_params)  # type: ignore
    elif extraction_schema is not None:
        # Use JSON schema approach
        transformed_schema = create_json_schema_with_source(
            extraction_schema, enrichment_class
        )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction_result",
                "schema": transformed_schema,
            },
        }

        request_params = _build_request_params(
            llm_model=llm_model,
            extraction_prompt=extraction_prompt,
            document=document,
            response_format=response_format,
            verbalized_confidence=verbalized_confidence,
        )

        response = litellm_client.chat.completions.create(**request_params)  # type: ignore
    else:
        raise ValueError("Must provide either extraction_model or extraction_schema.")

    return _process_llm_response(
        response=response,
        document=document,
        llm_model=llm_model,
        probability_aggregator=probability_aggregator,
        verbalized_confidence=verbalized_confidence,
    )


__all__ = [
    "groundit",
    "create_model_with_source",
    "add_confidence_scores",
    "FieldWithSourceAndConfidence",
    "average_probability_aggregator",
    "joint_probability_aggregator",
]
