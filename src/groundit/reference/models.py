from pydantic import Field, BaseModel
from typing import TypeVar, Generic


T = TypeVar("T")

description = """
The exact place in the source text from which the value was extracted OR inferred.
"""


class FieldWithSource(BaseModel, Generic[T]):
    """
    A generic container that wraps a field's value to add source tracking.

    This model holds the extracted `value` while preserving its original type,
    and includes an optional `source_quote` to store the exact text from
    which the value was extracted.
    """

    value: T = Field(description="The extracted value, preserving the original type.")
    source_quote: str = Field(description=description)


class FieldWithSourceAndConfidence(FieldWithSource[T]):
    """
    A generic container that extends FieldWithSource to add verbalized confidence scores.

    This model includes all fields from FieldWithSource plus verbalized confidence
    scores for both the extracted value and source quote, rated on a scale from 0 to 1.
    """

    value_confidence: float = Field(
        description="Verbalized confidence score for the extracted value, ranging from 0 to 1.",
        ge=0.0,
        le=1.0,
    )
    source_quote_confidence: float = Field(
        description="Verbalized confidence score for the source quote, ranging from 0 to 1.",
        ge=0.0,
        le=1.0,
    )
