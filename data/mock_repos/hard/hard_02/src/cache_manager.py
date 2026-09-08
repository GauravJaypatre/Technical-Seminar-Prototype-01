import time
from src.storage import CacheNode

class TTLLRUCache:
    def __init__(self, capacity: int, default_ttl: float = 60.0):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.index = {}

    def put(self, key: str, value, ttl: float = None):
        if ttl is None:
            ttl = self.default_ttl
        node = CacheNode(key, value, ttl)
        self.index[key] = node

    def get(self, key: str):
        if key not in self.index:
            return None
        node = self.index[key]
        if node.is_expired(time.time()):
            del self.index[key]
            return None
        return node.value

    # BUG: contains_key checks index directly without checking expiration!
    def contains_key(self, key: str) -> bool:
        return key in self.index
