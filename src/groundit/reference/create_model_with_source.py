from typing import Type, Union, get_origin, get_args
import types
from pydantic import BaseModel, create_model, Field
from groundit.reference.models import FieldWithSource


def create_source_model(model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Dynamically creates a new Pydantic model for source tracking.

    This function transforms a given Pydantic model into a new one where each
    leaf field is replaced by a `FieldWithSource` generic model. This allows for
    tracking the original text (`source_quote`) for each extracted value while
    preserving the original field's type and description.

    - Leaf fields are converted to `FieldWithSource[OriginalType]`.
    - Nested Pydantic models are recursively transformed.
    - Lists and Unions are traversed to transform their inner types.
    - Field descriptions from the original model are preserved.

    Args:
        model: The original Pydantic model class to transform.

    Returns:
        A new Pydantic model class with source tracking capabilities.
    """
    
    def _transform_type(original_type: Type) -> Type:
        """Recursively transforms a type annotation."""
        origin = get_origin(original_type)

        if origin:  # Handles generic types like list, union, etc.
            args = get_args(original_type)
            transformed_args = tuple(_transform_type(arg) for arg in args)
            
            # Handle Python 3.10+ UnionType (str | None syntax) 
            if isinstance(original_type, types.UnionType):
                return Union[transformed_args]
            
            return origin[transformed_args]

        # Handle nested Pydantic models
        if isinstance(original_type, type) and issubclass(original_type, BaseModel):
            return create_source_model(original_type)

        # Handle NoneType for optional fields
        if original_type is type(None):
            return type(None)

        # Base case: for leaf fields, wrap in FieldWithSource
        return FieldWithSource[original_type]

    transformed_fields = {}
    for field_name, field_info in model.model_fields.items():
        new_type = _transform_type(field_info.annotation)
        
        # Create a new Field, preserving the original description and default value
        new_field = Field(
            description=field_info.description,
            default=field_info.default if not field_info.is_required() else ...
        )
        transformed_fields[field_name] = (new_type, new_field)

    source_model_name = f"{model.__name__}WithSource"
    return create_model(source_model_name, **transformed_fields, __base__=BaseModel)

