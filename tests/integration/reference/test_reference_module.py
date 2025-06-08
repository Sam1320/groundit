import json
import pytest
from groundit.reference.add_source_spans import add_source_spans
from groundit.reference.create_model_with_source import create_source_model
from pydantic import BaseModel, Field
from datetime import date


JSON_EXTRACTION_SYSTEM_PROMPT = """
Extract data from the following document based on the JSON schema.
Return null if the document does not contain information relevant to schema.
Return only the JSON with no explanation text.
"""

DOCUMENT = """
<logo>MVZ im Fürstenberg-Karree Berlin<logo>

MVZ im Fürstenberg-Karree Berlin - Alexanderplatz 5 - 10178 Berlin

Dr. med. Markus Schneider-Burrus
Medicum Zentrum Berlin
Friedrichstraße 250
10245 Berlin

E-Nr: *D2020-008873*

Name: Müller
Vorname: Moritz
Geburtsdatum: 06.03.1998

23.06.2020

Probeneingang: 23.06.2020
Probenmaterial/klinische Angaben:
V. a. malignes Melanom, DD Nävus, Nagel Dig. V rechts, 1. Nagelbiopsie, 2. Nagelbett.

Makroskopie:
In zwei Gefäßen übersandt: 1., 2. Jeweils eine runde Probe von 3 mm Durchmesser. Vollständige Einbettung, 2 Paraffinblöcke, Schnittstufen, HE, PAS-Färbung, Immunhistologie.

Mikroskopie:
1. Kompakter, regelhaft aufgebauter Nagel, lediglich fokal gering zerfasert und mit geringer Parakeratose. Keine abnorme Pigmentierung. Das miterfasste Epithel atypiefrei, keine melanozytären Nester.
Sox 10: Negativ.
2. Weiterer kompakter Nagel und atypiefreies Nagelbett.
PAS-Färbung: Kein Nachweis von Pilzhyphen oder -sporen.
Sox 10: Vereinzelte, regelhafte Melanozyten in der Junktionszone.

Diagnosen 1. + 2.): überwiegend regelhafter Nagel und tumorfreies Nagelbett
Lokalisation: Dig. V rechts Nagel und Nagelbett

Beide Proben wurden vollständig aufgearbeitet und immunhistologisch untersucht, dabei kein Nachweis einer melanozytären Neoplasie, kein Anhalt für Malignität.

Priv.-Doz. Dr. med. Lars Moravec

Dieser Befund wurde elektronisch erstellt und freigegeben und ist auch ohne Unterschrift gültig.

<page_number>1 von 2<page_number>
"""


@pytest.fixture
def openai_client(openai_api_key):
    """Create OpenAI client with API key."""
    try:
        import openai
    except ImportError:
        pytest.skip("OpenAI package not installed")
    
    return openai.OpenAI(api_key=openai_api_key)



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

    def _validate_source_spans(self, data: dict | list, source_text: str):
        """
        Recursively traverse an extraction result and verify that each `source_span`
        correctly points to its corresponding `source_quote` in the original text.
        """
        if isinstance(data, dict):
            # Check if this is a field with source tracking information
            if 'source_span' in data and 'source_quote' in data and data['source_quote'] is not None:
                span = data['source_span']
                quote = data['source_quote']
                
                # Check if a valid span was found
                if span != [-1, -1]:
                    extracted_text = source_text[span[0]:span[1]]
                    assert extracted_text == quote, \
                        f"Span validation failed. Expected '{quote}', got '{extracted_text}'"

            # Recurse into dictionary values
            for value in data.values():
                self._validate_source_spans(value, source_text)
        
        elif isinstance(data, list):
            # Recurse into list items
            for item in data:
                self._validate_source_spans(item, source_text)

    def test_extraction_and_grounding(self, openai_client):
        """
        Test the full data extraction and grounding pipeline.
        
        This test ensures that a Pydantic model can be transformed for source tracking,
        used to extract data from a document via an LLM, and that the resulting data
        is correctly enriched with valid source spans.
        """
        # 1. Create a "source-aware" version of the Pydantic model
        response_model_with_source = create_source_model(Patient)

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
                    "content": DOCUMENT
                }
            ],
            logprobs=True,
            response_format=response_model_with_source
        )
        content = response.choices[0].message.content
        
        # 3. Add character-level source spans to the extraction result
        parsed_content = json.loads(content)
        enriched_result = add_source_spans(parsed_content, DOCUMENT)
        
        # 4. Validate that the generated spans correctly match the quotes
        self._validate_source_spans(enriched_result, DOCUMENT)
        
        # 5. Validate that the final result can be loaded back into the Pydantic model
        final_instance = response_model_with_source.model_validate(enriched_result)
        assert final_instance is not None
    
