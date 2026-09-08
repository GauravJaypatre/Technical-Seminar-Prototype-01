import pytest
from src.version_parser import tokenize_version

def test_basic_version():
    assert tokenize_version("1.2.3") == [1, 2, 3]

def test_empty_string():
    assert tokenize_version("") == []

def test_hyphenated_non_version():
    # FAILS ON UNPATCHED CODE: unescaped dot matches '-' splitting "10-20-30" into individual chars
    assert tokenize_version("10.20.30") == [10, 20, 30]

def test_non_dot_delimiters():
    # FAILS ON UNPATCHED CODE
    assert tokenize_version("100.200") == [100, 200]
