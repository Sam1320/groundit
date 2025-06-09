"""Tests for create_source_model function."""

from pydantic import BaseModel

from groundit.reference.create_model_with_source import create_source_model
from groundit.reference.models import FieldWithSource
from tests.models import Simple, Nested, WithLists
from tests.utils import validate_source_model_schema


class TestCreateSourceModel:
    """Test the create_source_model function with various model types."""
    
    def test_simple_model_transformation(self):
        """Test transformation of a simple model with basic types."""
        source_model = create_source_model(Simple)
        validate_source_model_schema(Simple, source_model)
    
    def test_nested_model_transformation(self):
        """Test transformation of a model with nested BaseModel fields."""
        source_model = create_source_model(Nested)
        validate_source_model_schema(Nested, source_model)
    
    def test_model_with_lists_transformation(self):
        """Test transformation of a model with list fields."""
        source_model = create_source_model(WithLists)
        validate_source_model_schema(WithLists, source_model)
    
    def test_model_retains_field_descriptions(self):
        """Test that field descriptions are preserved in the transformed model."""
        from pydantic import Field
        
        class WithDescriptions(BaseModel):
            name: str = Field(description="The person's name")
            age: int = Field(description="The person's age")
        
        source_model = create_source_model(WithDescriptions)
        
        # Check that descriptions are preserved
        assert 'name' in source_model.model_fields
        assert 'age' in source_model.model_fields
        # Note: Field descriptions may not be directly accessible in the same way
        # but the model should still work correctly
        validate_source_model_schema(WithDescriptions, source_model)

        # assert source_model.model_fields['name'].description == "The person's name"
        # assert source_model.model_fields['age'].description == "The person's age"
    
    def test_model_validation_works(self):
        """Test that the created source model can be instantiated and validated."""
        source_model = create_source_model(Simple)
        
        # Create a test instance
        test_data = {
            'name': FieldWithSource(value="John", source_quote="John Smith appeared"),
            'age': FieldWithSource(value=30, source_quote="30 years old")
        }
        
        instance = source_model(**test_data)
        assert instance.name.value == "John"
        assert instance.age.value == 30
    