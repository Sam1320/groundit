"""Test utilities for groundit confidence scoring."""

from typing import Type, Union
from pydantic import BaseModel, create_model

from groundit.reference.main import FieldWithSource


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
    but with leaf fields replaced by FieldWithSource.
    
    This is a test utility function for validating that create_source_model works correctly.
    
    Args:
        original_model: The original Pydantic model
        source_model: The transformed model with FieldWithSource fields
        
    Raises:
        AssertionError: If the models don't have the expected structure
    """
    # Check that field names are preserved
    original_fields = set(original_model.model_fields.keys())
    source_fields = set(source_model.model_fields.keys())
    assert original_fields == source_fields, f"Field names should match: {original_fields} vs {source_fields}"
    
    # Check each field
    for field_name in original_fields:
        original_field = original_model.model_fields[field_name]
        source_field = source_model.model_fields[field_name]
        
        original_type = original_field.annotation
        source_type = source_field.annotation
        
        # Check if original field is a BaseModel (nested model)
        if isinstance(original_type, type) and issubclass(original_type, BaseModel):
            # Source should also be a BaseModel (recursively transformed)
            if not (isinstance(source_type, type) and issubclass(source_type, BaseModel)):
                raise AssertionError(f"Nested field '{field_name}' should remain BaseModel in source, got {source_type}")
            # Recursively validate the nested model
            validate_source_model_schema(original_type, source_type)
            
        # Check if it's a list of BaseModels
        elif (hasattr(original_type, '__origin__') and original_type.__origin__ is list and
              len(original_type.__args__) == 1):
            inner_type = original_type.__args__[0]
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                # Source should be list of transformed BaseModel
                assert (hasattr(source_type, '__origin__') and source_type.__origin__ is list), \
                    f"List field '{field_name}' should remain list in source"
                source_inner_type = source_type.__args__[0]
                validate_source_model_schema(inner_type, source_inner_type)
            else:
                # List of primitives should become list of FieldWithSource
                expected_source_type = list[FieldWithSource]
                assert source_type == expected_source_type, \
                    f"Primitive list field '{field_name}' should become list[FieldWithSource], got {source_type}"
                    
        # Handle Union types (like str | None)
        elif hasattr(original_type, '__origin__') and original_type.__origin__ is Union:
            # Union types with primitives should become FieldWithSource
            assert source_type == FieldWithSource, \
                f"Union field '{field_name}' should become FieldWithSource, got {source_type}"
                
        else:
            # Leaf field should become FieldWithSource
            assert source_type == FieldWithSource, \
                f"Leaf field '{field_name}' should become FieldWithSource, got {source_type}" 