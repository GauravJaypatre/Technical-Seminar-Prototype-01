import pytest
from src.headers import CaseInsensitiveDict

def test_basic_storage():
    h = CaseInsensitiveDict({"Content-Type": "application/json"})
    assert h["content-type"] == "application/json"
    assert h["CONTENT-TYPE"] == "application/json"

def test_contains_case_insensitive():
    # FAILS ON UNPATCHED CODE
    h = CaseInsensitiveDict({"Authorization": "Bearer token123"})
    assert "authorization" in h
    assert "AUTHORIZATION" in h
    assert "Authorization" in h

def test_get_with_case_variations():
    # FAILS ON UNPATCHED CODE
    h = CaseInsensitiveDict({"X-Custom-Header": "foobar"})
    assert h.get("x-custom-header") == "foobar"
    assert h.get("X-CUSTOM-HEADER") == "foobar"
    assert h.get("non-existent", "default") == "default"
