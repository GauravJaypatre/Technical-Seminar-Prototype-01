import time

class CacheNode:
    def __init__(self, key: str, value, ttl: float):
        self.key = key
        self.value = value
        self.expiry = time.time() + ttl if ttl > 0 else float("inf")
        self.prev = None
        self.next = None

    def is_expired(self, current_time: float) -> bool:
        return current_time >= self.expiry
