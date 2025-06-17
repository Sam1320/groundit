from pydantic import BaseModel, Field
from datetime import date
from groundit import create_model_with_source, add_source_spans, get_confidence_scores, average_probability_aggregator
from groundit.confidence.confidence_extractor import add_confidence_scores
import openai
from rich import print, print_json
from rich.pretty import pprint
import json

model = "gpt-4.1"

class Patient(BaseModel):
    """
    A simplified model representing a patient resource,
    with inline definitions for Identifier and HumanName.
    """
    # name: list[HumanName] = Field(description="A name associated with the patient.")
    first_name: str = Field(description="The part of a name that links to the genealogy. In some cultures (e.g. Korean, Japanese, Vietnamese) this comes first.")
    last_name: str = Field(description="Given names (not always 'first'). Includes middle names.")
    # gender: str = Field(description="The gender of the patient. [male, female, other]")
    birthDate: date = Field(description="The date of birth for the individual.")
    # insurance_number: str = Field(description="The insurance number of the patient.")


JSON_EXTRACTION_SYSTEM_PROMPT = """
Extract data from the following document based on the JSON schema.
Return null if the document does not contain information relevant to schema.
If the information is present implicitly, fill the source field with the text that contains the information.
Return only the JSON with no explanation text.
"""

def extract_from_text(
    model: str,
    document: str,
    response_format: BaseModel,
) -> dict:
    """
    Extract structured data from text using Gemini model.
    
    Args:
        text: Text to extract data from
        json_schema: Schema describing the expected JSON structure
        
    Returns:
        Extracted JSON data
    """
    messages = [
        {
            "role": "system",
            "content": JSON_EXTRACTION_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": document
        }
    ]

        # 2. Use the transformed model to extract data from the document
    response = openai.beta.chat.completions.parse(
        model=model,
        messages=messages,
        logprobs=True,
        response_format=response_format
    )
    # logprobs = response.choices[0].logprobs.content
    # content = response.choices[0].message.content
    return response

# file_path = "data/nail_biopsy_2020.txt"
file_path = "tests/integration/data/example_doc.txt"
with open(file_path, "r") as file:
    document = file.read()

response_format = create_model_with_source(Patient)

# extract the data
result = extract_from_text(model=model, document=document, response_format=response_format)

content = result.choices[0].message.content

print_json(content)
print("-" * 100)

content_dict = json.loads(content)

# result_with_spans = add_source_spans(json.loads(content), document)
# pprint(result_with_spans, expand_all=True)
# print("-" * 100)

logprobs = result.choices[0].logprobs.content
# confidence_scores = get_confidence_scores(json_string_tokens=logprobs, aggregator=average_probability_aggregator)
# pprint(confidence_scores, expand_all=True)

result_with_confidence = add_confidence_scores(
    extraction_result=content_dict,
    tokens=logprobs,
    aggregator=average_probability_aggregator
)
pprint(result_with_confidence, expand_all=True)


result_with_spans = add_source_spans(result_with_confidence, document)
pprint(result_with_spans, expand_all=True)