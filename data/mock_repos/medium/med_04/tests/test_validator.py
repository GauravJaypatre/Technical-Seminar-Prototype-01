import pytest
from src.validator import validate_instance, SchemaValidationError

def test_valid_primitive():
    assert validate_instance("hello", {"type": "string"}) is True
    assert validate_instance(42, {"type": "integer"}) is True

def test_invalid_primitive():
    with pytest.raises(SchemaValidationError):
        validate_instance("not an int", {"type": "integer"})

def test_array_items_invalid_element():
    # FAILS ON UNPATCHED CODE
    schema = {
        "type": "array",
        "items": {"type": "integer"}
    }
    with pytest.raises(SchemaValidationError):
        validate_instance([1, 2, "three", 4], schema)

def test_array_items_valid():
    schema = {
        "type": "array",
        "items": {"type": "string"}
    }
    assert validate_instance(["apple", "banana"], schema) is True
