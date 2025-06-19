"""Integration tests for the complete groundit pipeline.

This module tests the full end-to-end workflow of using groundit
for data extraction with both Pydantic models and JSON schemas.
"""

import json
import pytest
from datetime import date
from pydantic import BaseModel, Field
from rich.pretty import pprint

from groundit import (
    create_model_with_source,
    create_json_schema_with_source, 
    add_source_spans,
    add_confidence_scores,
    average_probability_aggregator
)
from tests.utils import validate_source_spans


JSON_EXTRACTION_SYSTEM_PROMPT = """
Extract data from the following document based on the JSON schema.
Return null if the document does not contain information relevant to schema.
If the information is present implicitly, fill the source field with the text that contains the information.
Return only the JSON with no explanation text.
"""



class Patient(BaseModel):
    """Simple patient model for testing extraction."""
    first_name: str = Field(description="The given name of the patient")
    last_name: str = Field(description="The family name of the patient")
    birthDate: date = Field(description="The date of birth for the individual")


@pytest.mark.integration
@pytest.mark.slow
class TestGrounditPipeline:
    """Integration tests for the complete groundit pipeline."""
    
    @pytest.mark.parametrize("model", ["gpt-4o-mini"])
    def test_pydantic_model_full_pipeline(self, openai_client, test_document, model):
        """
        Test the complete groundit pipeline using Pydantic models.
        
        This test verifies:
        1. Model transformation with create_model_with_source
        2. LLM extraction with transformed model
        3. Confidence score addition
        4. Source span addition
        5. Validation of final enriched result
        """
        # 1. Transform the Pydantic model to include source tracking
        patient_with_source = create_model_with_source(Patient)
        
        # 2. Extract data using the transformed model
        response = openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": JSON_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": test_document}
            ],
            logprobs=True,
            response_format=patient_with_source
        )
        
        # Parse the response
        content = response.choices[0].message.content
        extraction_result = json.loads(content)
        tokens = response.choices[0].logprobs.content
        
        # 3. Add confidence scores
        result_with_confidence = add_confidence_scores(
            extraction_result=extraction_result,
            tokens=tokens,
            aggregator=average_probability_aggregator
        )
        
        # 4. Add source spans
        final_result = add_source_spans(result_with_confidence, test_document)
        
        # 5. Validate the complete pipeline result
        # Validate structure can be loaded back into the model
        validated_instance = patient_with_source.model_validate(final_result)
        assert validated_instance is not None
        
        # Validate that source spans are correct
        validate_source_spans(final_result, test_document) 

        # Validate confidence scores exist and are valid probabilities
        assert 0 < final_result["first_name"]["value_confidence"] <= 1.0
        assert 0 < final_result["last_name"]["value_confidence"] <= 1.0
        assert 0 < final_result["birthDate"]["value_confidence"] <= 1.0
        assert 0 < final_result["first_name"]["source_quote_confidence"] <= 1.0
        assert 0 < final_result["last_name"]["source_quote_confidence"] <= 1.0
        assert 0 < final_result["birthDate"]["source_quote_confidence"] <= 1.0
        

        print("="*50)
        pprint(final_result, expand_all=True)
    
    @pytest.mark.parametrize("model", ["gpt-4o-mini"])
    def test_json_schema_full_pipeline(self, openai_client, test_document, model):
        """
        Test the complete groundit pipeline using JSON schemas.
        
        This test verifies:
        1. JSON schema transformation with create_json_schema_with_source
        2. LLM extraction with transformed schema
        3. Confidence score addition
        4. Source span addition
        5. Validation that results match Pydantic model approach
        """
        # 1. Transform the JSON schema to include source tracking
        original_schema = Patient.model_json_schema()
        transformed_schema = create_json_schema_with_source(original_schema)
        
        # 2. Extract data using the transformed JSON schema
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JSON_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": test_document}
            ],
            logprobs=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "patient_extraction",
                    "schema": transformed_schema
                }
            }
        )
        
        # Parse the response
        content = response.choices[0].message.content
        extraction_result = json.loads(content)
        tokens = response.choices[0].logprobs.content
        
        # 3. Add confidence scores
        result_with_confidence = add_confidence_scores(
            extraction_result=extraction_result,
            tokens=tokens,
            aggregator=average_probability_aggregator
        )
        
        # 4. Add source spans
        final_result = add_source_spans(result_with_confidence, test_document)
        
        # 5. Validate the complete pipeline result
        # Validate that source spans are correct
        validate_source_spans(final_result, test_document)
        
        # Validate basic extraction accuracy
        assert "moritz" in final_result["first_name"]["value"].lower()
        assert "müller" in final_result["last_name"]["value"].lower()
        assert final_result["birthDate"]["value"] == "1998-03-06"
        
        # Validate confidence scores exist and are valid probabilities
        assert 0 < final_result["first_name"]["value_confidence"] <= 1.0
        assert 0 < final_result["last_name"]["value_confidence"] <= 1.0
        assert 0 < final_result["birthDate"]["value_confidence"] <= 1.0
        assert 0 < final_result["first_name"]["source_quote_confidence"] <= 1.0
        assert 0 < final_result["last_name"]["source_quote_confidence"] <= 1.0
        assert 0 < final_result["birthDate"]["source_quote_confidence"] <= 1.0
        
        # Validate the result can be loaded into the Pydantic model
        patient_with_source = create_model_with_source(Patient)
        validated_instance = patient_with_source.model_validate(final_result)
        assert validated_instance is not None
        
        print("\n" + "="*50)
        print("JSON SCHEMA PIPELINE RESULT")
        print("="*50)
        pprint(final_result, expand_all=True)
    
    def test_pipeline_consistency(self, openai_client, test_document):
        """
        Test that Pydantic model and JSON schema approaches produce equivalent results.
        
        This test verifies that both transformation approaches (runtime model vs schema)
        produce structurally identical results when used in the complete pipeline.
        """
        model = "gpt-4o-mini"
        
        # Get results from both approaches
        # Pydantic approach
        patient_with_source = create_model_with_source(Patient)
        pydantic_response = openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": JSON_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": test_document}
            ],
            logprobs=True,
            response_format=patient_with_source
        )
        
        # JSON schema approach  
        original_schema = Patient.model_json_schema()
        transformed_schema = create_json_schema_with_source(original_schema)
        schema_response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JSON_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": test_document}
            ],
            logprobs=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "patient_extraction",
                    "schema": transformed_schema
                }
            }
        )
        
        # Process both results through the same pipeline
        pydantic_result = json.loads(pydantic_response.choices[0].message.content)
        schema_result = json.loads(schema_response.choices[0].message.content)
        
        # Both should have the same structure (keys and nested structure)
        assert set(pydantic_result.keys()) == set(schema_result.keys())
        
        # Both should be valid according to the source model
        patient_with_source.model_validate(pydantic_result)
        patient_with_source.model_validate(schema_result)
        
        print("\n" + "="*50)
        print("PIPELINE CONSISTENCY VERIFICATION")
        print("="*50)
        print("Both approaches produce structurally compatible results")