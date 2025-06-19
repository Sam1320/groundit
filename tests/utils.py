"""Test utilities for groundit confidence scoring."""

from typing import Type
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


def validate_source_model_schema(original_model: type[BaseModel], model_with_source: type[BaseModel]) -> None:
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
    fields_with_source = set(model_with_source.model_fields.keys())
    assert original_fields == fields_with_source, f"Field names should match: {original_fields} vs {fields_with_source}"
    
    for field_name in original_fields:
        original_field = original_model.model_fields[field_name]
        field_with_source = model_with_source.model_fields[field_name]
        
        # Check that description is preserved
        assert original_field.description == field_with_source.description, \
            f"Field '{field_name}' description should be preserved. " \
            f"Got '{field_with_source.description}' instead of '{original_field.description}'"

        def validate_type(original_type: Type, type_with_source: Type, field_name: str):
            """Recursive helper to validate types."""
            if hasattr(original_type, '__origin__'):
                # Handle list types
                if original_type.__origin__ is list:
                    assert hasattr(type_with_source, '__origin__') and type_with_source.__origin__ is list, \
                        f"Type for field '{field_name}' should be a list in source model."
                    validate_type(original_type.__args__[0], type_with_source.__args__[0], field_name)
                    return

            # Handle nested Pydantic models
            if isinstance(original_type, type) and issubclass(original_type, BaseModel):
                assert isinstance(type_with_source, type) and issubclass(type_with_source, BaseModel), \
                    f"Nested model field '{field_name}' should remain a BaseModel."
                validate_source_model_schema(original_type, type_with_source)
                return

            # Base case: Leaf fields should become FieldWithSource
            assert type_with_source is FieldWithSource[original_type], \
                f"Leaf field '{field_name}' of type {original_type} should become FieldWithSource[{original_type}], but got {type_with_source}."

        validate_type(original_field.annotation, field_with_source.annotation, field_name)


def validate_source_spans(data: dict | list, source_text: str):
    """
    Recursively validate that source spans correctly match source quotes.
    
    Args:
        data: The data structure with potential source spans
        source_text: The original source document
    """
    if isinstance(data, dict):
        if 'source_span' in data and 'source_quote' in data and data['source_quote'] is not None:
            span = data['source_span']
            quote = data['source_quote']
            
            if span != [-1, -1]:
                extracted_text = source_text[span[0]:span[1]]
                assert extracted_text == quote, \
                    f"Span validation failed. Expected '{quote}', got '{extracted_text}'"
        
        for value in data.values():
            validate_source_spans(value, source_text)
    
    elif isinstance(data, list):
        for item in data:
            validate_source_spans(item, source_text) 