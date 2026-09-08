import pytest
from src.config_merger import merge_configs

def test_flat_merge():
    base = {"a": 1, "b": 2}
    over = {"b": 3, "c": 4}
    assert merge_configs(base, over) == {"a": 1, "b": 3, "c": 4}

def test_nested_merge_preserves_sibling_keys():
    # FAILS ON UNPATCHED CODE
    base = {"database": {"host": "localhost", "port": 5432, "timeout": 30}}
    override = {"database": {"port": 5433}}
    res = merge_configs(base, override)
    assert res["database"]["port"] == 5433
    assert res["database"]["host"] == "localhost"
    assert res["database"]["timeout"] == 30
