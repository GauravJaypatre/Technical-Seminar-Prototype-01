class SchemaValidationError(Exception):
    pass

TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

def validate_instance(instance, schema: dict) -> bool:
    expected_type_name = schema.get("type")
    if expected_type_name:
        expected_type = TYPE_MAP.get(expected_type_name)
        if expected_type and not isinstance(instance, expected_type):
            raise SchemaValidationError(f"Expected {expected_type_name}, got {type(instance).__name__}")
    
    # BUG: If type is array and 'items' is present in schema, items are not checked!
    return True
