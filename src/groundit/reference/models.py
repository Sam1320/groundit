from pydantic import Field, BaseModel
from typing import TypeVar, Generic


T = TypeVar('T')

class FieldWithSource(BaseModel, Generic[T]):
    """
    A generic container that wraps a field's value to add source tracking.
    
    This model holds the extracted `value` while preserving its original type,
    and includes an optional `source_quote` to store the exact text from
    which the value was extracted.
    """
    value: T = Field(description="The extracted value, preserving the original type.")
    source_quote: str | None = Field(
        default=None,
        description="The exact sentence or phrase from the source text from which the value was extracted."
    )