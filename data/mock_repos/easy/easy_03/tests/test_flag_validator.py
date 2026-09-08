import pytest
from src.flag_validator import validate_flag

def test_valid_flag():
    assert validate_flag("--verbose", {"--verbose", "--quiet"}) is True
    assert validate_flag("--VERBOSE", {"--verbose", "--quiet"}) is True

def test_invalid_flag():
    assert validate_flag("--debug", {"--verbose", "--quiet"}) is False

def test_none_flag_handled_safely():
    # FAILS ON UNPATCHED CODE
    assert validate_flag(None, {"--verbose", "--quiet"}) is False
