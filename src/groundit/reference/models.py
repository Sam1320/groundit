from typing import Any
from pydantic import BaseModel, Field


class FieldWithSource(BaseModel):
    """
    A field with a source.
    """
    value: Any = Field(description="The value of the field.")
    source_quote: str = Field(description="The exact sentence or phrase from the source text from which the value was extracted.")