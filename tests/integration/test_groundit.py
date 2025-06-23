"""Integration tests for the complete groundit pipeline.

This module tests the full end-to-end workflow of using groundit
for data extraction with both Pydantic models and JSON schemas.
"""

import pytest
from pydantic import BaseModel, Field
from rich.pretty import pprint

from groundit import (
    groundit,
    create_model_with_source
)
from tests.utils import validate_source_spans


class Patient(BaseModel):
    """Simple patient model for testing extraction."""
    first_name: str = Field(description="The given name of the patient")
    last_name: str = Field(description="The family name of the patient")
    birthDate: str = Field(description="The date of birth for the individual")


@pytest.mark.integration
@pytest.mark.slow
class TestGrounditPipeline:
    """Integration tests for the complete groundit pipeline."""
    
    def test_pydantic_model_full_pipeline(self, openai_client, test_document):
        """
        Test the complete groundit pipeline using Pydantic models.
        
        This test verifies the unified groundit() function works with Pydantic models.
        """
        # Use the unified groundit function
        final_result = groundit(document=test_document, extraction_schema=Patient)
        
        # Validate the complete pipeline result
        # Validate structure can be loaded back into the model
        patient_with_source = create_model_with_source(Patient)
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
    
    def test_json_schema_full_pipeline(self, openai_client, test_document):
        """
        Test the complete groundit pipeline using JSON schemas.
        
        This test verifies the unified groundit() function works with JSON schemas.
        """
        # Use the unified groundit function with JSON schema
        original_schema = Patient.model_json_schema()
        final_result = groundit(document=test_document, extraction_schema=original_schema)
        
        # Validate the complete pipeline result
        # Validate that source spans are correct
        validate_source_spans(final_result, test_document)
        
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
        produce structurally identical results when used in the groundit() function.
        """
        
        # Get results from both approaches using the unified function
        # Pydantic approach
        pydantic_result = groundit(
            document=test_document,
            extraction_schema=Patient
        )
        
        # JSON schema approach  
        original_schema = Patient.model_json_schema()
        schema_result = groundit(document=test_document, extraction_schema=original_schema)
        
        # Both should have the same structure (keys and nested structure)
        assert set(pydantic_result.keys()) == set(schema_result.keys())
        
        # Both should be valid according to the source model
        patient_with_source = create_model_with_source(Patient)
        patient_with_source.model_validate(pydantic_result)
        patient_with_source.model_validate(schema_result)