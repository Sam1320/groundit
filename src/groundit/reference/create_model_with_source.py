from typing import Type, Union
from pydantic import BaseModel, create_model, Field
from groundit.reference.models import FieldWithSource


def create_source_model(model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a model with source tracking from a given Pydantic model.
    This function transforms the model to include source tracking while preserving
    field descriptions.
    - Leaf fields are converted to `FieldWithSource`.
    - Nested Pydantic models are recursively transformed.
    - Field descriptions from the original model are preserved in the new model.
    Args:
        model: The original Pydantic model to transform.
    Returns:
        A new model class with source tracking capabilities.
    """
    
    def transform_type(original_type: Type) -> Type:
        """Recursively transforms the type annotation."""
        if hasattr(original_type, '__origin__'):
            if original_type.__origin__ is Union:
                # Handle Union types, e.g., str | None
                new_args = tuple(transform_type(arg) for arg in original_type.__args__)
                return Union[new_args]
            
            if original_type.__origin__ is list:
                # Handle list types, e.g., list[HumanName]
                inner_type = original_type.__args__[0]
                transformed_inner = transform_type(inner_type)
                return list[transformed_inner]

        # Handle nested Pydantic models
        if isinstance(original_type, type) and issubclass(original_type, BaseModel):
            return create_source_model(original_type)

        # Handle NoneType for optional fields
        if original_type is type(None):
            return type(None)

        # Base case: for leaf fields, use FieldWithSource
        return FieldWithSource

    transformed_fields = {}
    for field_name, field_info in model.model_fields.items():
        new_type = transform_type(field_info.annotation)
        
        # Create a new Field, preserving the original description and default value
        new_field = Field(
            description=field_info.description,
            default=field_info.default if not field_info.is_required() else ...
        )
        transformed_fields[field_name] = (new_type, new_field)

    source_model_name = f"{model.__name__}WithSource"
    return create_model(source_model_name, **transformed_fields, __base__=BaseModel)

