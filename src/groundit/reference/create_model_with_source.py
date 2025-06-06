from typing import Type, Union
from pydantic import BaseModel, create_model
from groundit.reference.models import FieldWithSource

def create_source_model(model: BaseModel) -> Type[BaseModel]:
    """
    Create a ModelWithSource from a given model.
    Convert every leaf field to a FieldWithSource field.
    
    Args:
        model: The original Pydantic model to transform
        
    Returns:
        A new model class with the same structure but FieldWithSource types for leaf fields
    """
    def transform_field_type(field_info) -> tuple:
        """Transform a field's type annotation to use FieldWithSource for leaf fields."""
        field_type = field_info.annotation
        
        # Handle Union types (like str | None)
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            # For union types, check if any are BaseModel subclasses
            union_args = field_type.__args__
            has_basemodel = any(
                isinstance(arg, type) and issubclass(arg, BaseModel) 
                for arg in union_args if arg is not type(None)
            )
            if has_basemodel:
                # Transform the BaseModel part
                non_none_args = [arg for arg in union_args if arg is not type(None)]
                if len(non_none_args) == 1 and isinstance(non_none_args[0], type) and issubclass(non_none_args[0], BaseModel):
                    transformed_model = create_source_model(non_none_args[0])
                    if type(None) in union_args:
                        return (Union[transformed_model, type(None)], field_info.default)
                    else:
                        return (transformed_model, field_info.default)
            
            # For other union types (like str | None), treat as leaf
            return (FieldWithSource, field_info.default)
        
        # Handle list types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            list_args = field_type.__args__
            if len(list_args) == 1:
                inner_type = list_args[0]
                if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                    # List of BaseModels - transform the inner type
                    transformed_inner = create_source_model(inner_type)
                    return (list[transformed_inner], field_info.default)
                else:
                    # List of leaf types - keep as list but content becomes FieldWithSource
                    return (list[FieldWithSource], field_info.default)
        
        # Check if it's a BaseModel subclass (nested model) - recursively transform
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return (create_source_model(field_type), field_info.default)
        
        # For leaf fields, use FieldWithSource
        return (FieldWithSource, field_info.default)
    
    # Transform all fields
    transformed_fields = {}
    for field_name, field_info in model.model_fields.items():
        transformed_fields[field_name] = transform_field_type(field_info)
    
    # Create new model with transformed fields
    source_model_name = f"{model.__name__}WithSource"
    return create_model(source_model_name, **transformed_fields)

