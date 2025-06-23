import json
from typing import Any, Type, Optional
from pydantic import BaseModel
from openai import OpenAI

from groundit.confidence.confidence_extractor import add_confidence_scores, get_confidence_scores
from groundit.confidence.logprobs_aggregators import (
    AggregationFunction,
    average_probability_aggregator,
    joint_probability_aggregator,
    default_sum_aggregator
)
from groundit.reference.add_source_spans import add_source_spans
from groundit.reference.create_model_with_source import create_model_with_source, create_json_schema_with_source
from groundit.config import (
    DEFAULT_EXTRACTION_PROMPT,
    DEFAULT_LLM_MODEL,
    DEFAULT_PROBABILITY_AGGREGATOR
)


def groundit(
    document: str,
    extraction_schema: Type[BaseModel] | dict[str, Any],
    extraction_prompt: Optional[str] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    probability_aggregator: AggregationFunction = DEFAULT_PROBABILITY_AGGREGATOR,
    openai_client: Optional[OpenAI] = None
) -> dict[str, Any]:
    """
    Complete groundit pipeline for data extraction with confidence scores and source tracking.
    
    This function orchestrates the full groundit workflow:
    1. Transform schema to include source tracking
    2. Extract data using LLM with transformed schema
    3. Add confidence scores based on token probabilities
    4. Add source spans linking extracted values to document text
    
    Args:
        document: The source document to extract information from
        extraction_schema: Either a Pydantic model class or JSON schema dict
        extraction_prompt: System prompt for guiding the extraction (uses default if None)
        llm_model: OpenAI model to use for extraction
        probability_aggregator: Function to aggregate token probabilities into confidence scores
        openai_client: OpenAI client instance (creates default if None)
    
    Returns:
        Dictionary with extracted data enriched with confidence scores and source quotes
    """
    if openai_client is None:
        openai_client = OpenAI()
    
    if extraction_prompt is None:
        extraction_prompt = DEFAULT_EXTRACTION_PROMPT
    
    # Determine if we're working with a Pydantic model or JSON schema
    is_pydantic_model = isinstance(extraction_schema, type) and issubclass(extraction_schema, BaseModel)
    
    if is_pydantic_model:
        # Use Pydantic model approach
        model_with_source = create_model_with_source(extraction_schema)
        
        response = openai_client.beta.chat.completions.parse(
            model=llm_model,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": document}
            ],
            logprobs=True,
            response_format=model_with_source
        )
    else:
        # Use JSON schema approach
        transformed_schema = create_json_schema_with_source(extraction_schema)
        
        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": document}
            ],
            logprobs=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_result",
                    "schema": transformed_schema
                }
            }
        )
    
    # Parse the response
    content = response.choices[0].message.content
    extraction_result = json.loads(content)
    tokens = response.choices[0].logprobs.content
    
    # Add confidence scores
    result_with_confidence = add_confidence_scores(
        extraction_result=extraction_result,
        tokens=tokens,
        aggregator=probability_aggregator
    )
    
    # Add source spans
    final_result = add_source_spans(result_with_confidence, document)
    
    return final_result


__all__ = [
    "groundit",
    "get_confidence_scores",
    "add_confidence_scores",
    "add_source_spans",
    "create_model_with_source",
    "create_json_schema_with_source",
    "AggregationFunction",
    "average_probability_aggregator",
    "joint_probability_aggregator",
    "default_sum_aggregator",
    "DEFAULT_EXTRACTION_PROMPT",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_PROBABILITY_AGGREGATOR",
]