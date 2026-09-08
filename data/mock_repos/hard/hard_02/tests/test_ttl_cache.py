import pytest
import time
from src.cache_manager import TTLLRUCache

def test_put_and_get():
    cache = TTLLRUCache(capacity=5, default_ttl=10.0)
    cache.put("k1", "val1")
    assert cache.get("k1") == "val1"
    assert cache.contains_key("k1") is True

def test_contains_key_respects_expiration():
    # FAILS ON UNPATCHED CODE
    cache = TTLLRUCache(capacity=5, default_ttl=0.05)
    cache.put("ephemeral", "data")
    assert cache.contains_key("ephemeral") is True
    time.sleep(0.08)
    assert cache.contains_key("ephemeral") is False
