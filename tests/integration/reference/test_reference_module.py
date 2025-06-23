import json
from groundit.reference.add_source_spans import add_source_spans
from groundit.reference.create_model_with_source import create_model_with_source
from pydantic import BaseModel, Field
from datetime import date
from tests.utils import validate_source_spans


JSON_EXTRACTION_SYSTEM_PROMPT = """
Extract data from the following document based on the JSON schema.
Return null if the document does not contain information relevant to schema.
Return only the JSON with no explanation text.
"""




class HumanName(BaseModel):
    """A human name."""
    family: str = Field(description="The part of a name that links to the genealogy. In some cultures (e.g. Korean, Japanese, Vietnamese) this comes first.")
    given: list[str] = Field(description="Given names (not always 'first'). Includes middle names.")


class Patient(BaseModel):
    """A simplified model representing a patient resource."""
    name: list[HumanName] = Field(description="A name associated with the patient.")
    birthDate: date = Field(description="The date of birth for the individual.")


class TestReferenceModule:
    """
    Integration tests for the complete reference module pipeline.
    
    This test suite verifies the end-to-end process of:
    1. Transforming a Pydantic model to include source tracking.
    2. Using the transformed model for data extraction with an LLM.
    3. Enriching the extracted data with character-level source spans.
    4. Validating the correctness of the generated source spans.
    """


    def test_extraction_and_grounding(self, openai_client, test_document):
        """
        Test the full data extraction and grounding pipeline.
        
        This test ensures that a Pydantic model can be transformed for source tracking,
        used to extract data from a document via an LLM, and that the resulting data
        is correctly enriched with valid source spans.
        """
        # 1. Create a "source-aware" version of the Pydantic model
        response_model_with_source = create_model_with_source(Patient)

        # 2. Use the transformed model to extract data from the document
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": JSON_EXTRACTION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": test_document
                }
            ],
            logprobs=True,
            response_format=response_model_with_source
        )
        content = response.choices[0].message.content
        
        # 3. Add character-level source spans to the extraction result
        parsed_content = json.loads(content)
        enriched_result = add_source_spans(parsed_content, test_document)
        
        # 4. Validate that the generated spans correctly match the quotes
        validate_source_spans(enriched_result, test_document)
        
        # 5. Validate that the final result can be loaded back into the Pydantic model
        final_instance = response_model_with_source.model_validate(enriched_result)
        assert final_instance is not None
    
