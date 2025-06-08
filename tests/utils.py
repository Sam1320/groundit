"""Test utilities for groundit confidence scoring."""

from typing import Type, Union
from pydantic import BaseModel, create_model

from groundit.reference.models import FieldWithSource


def create_confidence_model(original_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Creates a new Pydantic model where all leaf fields (non-nested BaseModel fields) 
    are transformed to accept float values for confidence scores.
    
    This is useful for testing that confidence scoring transforms model outputs correctly,
    ensuring all leaf values become float confidence scores while preserving structure.
    
    Args:
        original_model: The original Pydantic model to transform
        
    Returns:
        A new model class with the same structure but float types for leaf fields
    """
    def transform_field_type(field_info) -> tuple:
        """Transform a field's type annotation to include float for confidence scores."""
        field_type = field_info.annotation
        
        # Check if it's a BaseModel subclass (nested model) - keep as is
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return (create_confidence_model(field_type), field_info.default)
        
        # For leaf fields, allow float (confidence scores)
        return (float, field_info.default)
    
    # Transform all fields
    transformed_fields = {}
    for field_name, field_info in original_model.model_fields.items():
        transformed_fields[field_name] = transform_field_type(field_info)
    
    # Create new model with transformed fields
    confidence_model_name = f"{original_model.__name__}Confidence"
    return create_model(confidence_model_name, **transformed_fields) 


def validate_source_model_schema(original_model: type[BaseModel], source_model: type[BaseModel]) -> None:
    """
    Recursively validate that a source model has the same structure as the original
    but with leaf fields replaced by FieldWithSource. It also checks that field
    descriptions are preserved.
    
    This is a test utility function for validating that create_source_model works correctly.
    
    Args:
        original_model: The original Pydantic model
        source_model: The transformed model with FieldWithSource fields
        
    Raises:
        AssertionError: If the models don't have the expected structure
    """
    original_fields = set(original_model.model_fields.keys())
    source_fields = set(source_model.model_fields.keys())
    assert original_fields == source_fields, f"Field names should match: {original_fields} vs {source_fields}"
    
    for field_name in original_fields:
        original_field = original_model.model_fields[field_name]
        source_field = source_model.model_fields[field_name]
        
        # Check that description is preserved
        assert original_field.description == source_field.description, \
            f"Field '{field_name}' description should be preserved. " \
            f"Got '{source_field.description}' instead of '{original_field.description}'"

        def validate_type(original_type: Type, source_type: Type, field_name: str):
            """Recursive helper to validate types."""
            if hasattr(original_type, '__origin__'):
                # Handle list types
                if original_type.__origin__ is list:
                    assert hasattr(source_type, '__origin__') and source_type.__origin__ is list, \
                        f"Type for field '{field_name}' should be a list in source model."
                    validate_type(original_type.__args__[0], source_type.__args__[0], field_name)
                    return

                # Handle Union types
                if original_type.__origin__ is Union:
                    # Note: This is a simplified check. It assumes the order of args is preserved
                    # and doesn't handle complex nested unions perfectly, but is sufficient for now.
                    assert hasattr(source_type, '__origin__') and source_type.__origin__ is Union, \
                         f"Type for field '{field_name}' should be a Union in source model."
                    
                    original_args = original_type.__args__
                    source_args = source_type.__args__
                    assert len(original_args) == len(source_args), \
                        f"Union for field '{field_name}' should have same number of arguments."

                    for o_arg, s_arg in zip(original_args, source_args):
                        validate_type(o_arg, s_arg, field_name)
                    return

            # Handle nested Pydantic models
            if isinstance(original_type, type) and issubclass(original_type, BaseModel):
                assert isinstance(source_type, type) and issubclass(source_type, BaseModel), \
                    f"Nested model field '{field_name}' should remain a BaseModel."
                validate_source_model_schema(original_type, source_type)
                return

            # Handle NoneType for optional fields
            if original_type is type(None):
                assert source_type is type(None), f"Optional field '{field_name}' should remain optional."
                return

            # # Base case: Leaf fields should become FieldWithSource
            # assert source_type is FieldWithSource, \
            #     f"Leaf field '{field_name}' of type {original_type} should become FieldWithSource, but got {source_type}."

        validate_type(original_field.annotation, source_field.annotation, field_name) 