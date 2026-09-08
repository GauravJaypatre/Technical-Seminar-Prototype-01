import pytest
from src.rate_limiter import SlidingWindowRateLimiter

def test_allows_under_limit():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10.0)
    assert limiter.allow_request(0.0) is True
    assert limiter.allow_request(1.0) is True
    assert limiter.allow_request(2.0) is False

def test_boundary_expiration():
    # FAILS ON UNPATCHED CODE
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10.0)
    assert limiter.allow_request(0.0) is True
    assert limiter.allow_request(5.0) is False
    assert limiter.allow_request(10.0) is True
