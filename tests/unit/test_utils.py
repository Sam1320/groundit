"""Tests for the validate_source_model_schema test utility function itself."""

import pytest
from pydantic import BaseModel

from groundit.reference.models import FieldWithSource
from tests.utils import validate_source_model_schema


class TestPositiveCases:
    """Test cases that should pass validation."""
    
    def test_simple_model_correctly_transformed(self):
        """Test that a simple model is correctly transformed."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class SimpleWithSource(BaseModel):
            name: FieldWithSource[str]
            age: FieldWithSource[int]
        
        # Should pass without raising
        validate_source_model_schema(Simple, SimpleWithSource)
    
    def test_nested_model_correctly_transformed(self):
        """Test that a nested model is correctly transformed."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class Nested(BaseModel):
            profile: Simple
            active: bool
        
        class SimpleWithSource(BaseModel):
            name: FieldWithSource[str]
            age: FieldWithSource[int]
            
        class NestedWithSource(BaseModel):
            profile: SimpleWithSource
            active: FieldWithSource[bool]
        
        # Should pass without raising
        validate_source_model_schema(Nested, NestedWithSource)


class TestFieldNameValidation:
    """Test cases for field name validation errors."""
    
    def test_wrong_field_names_rejected(self):
        """Test that models with wrong field names are rejected."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class SimpleWithSource(BaseModel):
            wrong_name: FieldWithSource[str]  # Should be 'name'
            age: FieldWithSource[int]
        
        with pytest.raises(AssertionError, match="Field names should match"):
            validate_source_model_schema(Simple, SimpleWithSource)
    
    def test_missing_field_rejected(self):
        """Test that models with missing fields are rejected."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class SimpleWithSource(BaseModel):
            name: FieldWithSource[str]
            # Missing 'age' field
        
        with pytest.raises(AssertionError, match="Field names should match"):
            validate_source_model_schema(Simple, SimpleWithSource)
    
    def test_extra_field_rejected(self):
        """Test that models with extra fields are rejected."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class SimpleWithSource(BaseModel):
            name: FieldWithSource[str]
            age: FieldWithSource[int]
            extra: FieldWithSource[str]  # Extra field not in original
        
        with pytest.raises(AssertionError, match="Field names should match"):
            validate_source_model_schema(Simple, SimpleWithSource)


class TestFieldTypeValidation:
    """Test cases for field type validation errors."""
    
    def test_wrong_leaf_field_type_rejected(self):
        """Test that leaf fields with wrong types are rejected."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class SimpleWithSource(BaseModel):
            name: str  # Should be FieldWithSource[
            age: FieldWithSource[int]
        
        with pytest.raises(AssertionError, match="should become FieldWithSource"):
            validate_source_model_schema(Simple, SimpleWithSource)
    
    def test_wrong_nested_structure_rejected(self):
        """Test that nested models with wrong structure are rejected."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class Nested(BaseModel):
            profile: Simple
            active: bool
        
        class SimpleWithSource(BaseModel):
            name: str  # Should be FieldWithSource
            age: FieldWithSource[int]
        
        class NestedWithSource(BaseModel):
            profile: SimpleWithSource  # Contains invalid nested structure
            active: FieldWithSource[bool]
        
        with pytest.raises(AssertionError, match="should become FieldWithSource"):
            validate_source_model_schema(Nested, NestedWithSource)
    
    def test_nested_field_wrong_base_model_structure_rejected(self):
        """Test that nested fields with wrong BaseModel structure are rejected."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class Nested(BaseModel):
            profile: Simple
            active: bool
        
        class NestedWithSource(BaseModel):
            profile: FieldWithSource  # Should be transformed Simple, not FieldWithSource
            active: FieldWithSource[bool]
        
        # FieldWithSource is technically a BaseModel, but has wrong field structure
        with pytest.raises(AssertionError, match="Field names should match"):
            validate_source_model_schema(Nested, NestedWithSource)


class TestListValidation:
    """Test cases for list type validation."""
    
    def test_primitive_lists_correctly_transformed(self):
        """Test that primitive lists are correctly transformed."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class WithLists(BaseModel):
            tags: list[str]
            profiles: list[Simple]
        
        class SimpleWithSource(BaseModel):
            name: FieldWithSource[str]
            age: FieldWithSource[int]
        
        class WithListsWithSource(BaseModel):
            tags: list[FieldWithSource[str]]  # Primitive list becomes list[FieldWithSource]
            profiles: list[SimpleWithSource]  # BaseModel list becomes list[TransformedBaseModel]
        
        # Should pass without raising
        validate_source_model_schema(WithLists, WithListsWithSource)
    
    def test_wrong_list_transformation_rejected(self):
        """Test that incorrect list transformations are rejected."""
        class Simple(BaseModel):
            name: str
            age: int
        
        class WithLists(BaseModel):
            tags: list[str]
            profiles: list[Simple]
        
        class SimpleWithSource(BaseModel):
            name: FieldWithSource[str]
            age: FieldWithSource[int]
        
        class WithListsWithSource(BaseModel):
            tags: list[str]  # Should be list[FieldWithSource]
            profiles: list[SimpleWithSource]
        
        with pytest.raises(AssertionError): #, match="should become list\\[FieldWithSource\\]"):
            validate_source_model_schema(WithLists, WithListsWithSource)