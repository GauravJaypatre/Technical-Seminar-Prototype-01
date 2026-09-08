from collections import deque

class SlidingWindowRateLimiter:
    """Rate limiter enforcing maximum requests within a sliding window in seconds."""
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def allow_request(self, current_time: float) -> bool:
        # BUG: uses (current_time - timestamp) > window_seconds instead of >=
        while self.timestamps and (current_time - self.timestamps[0]) > self.window_seconds:
            self.timestamps.popleft()

        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(current_time)
            return True
        return False
