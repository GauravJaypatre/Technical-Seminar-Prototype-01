from typing import List

def sample_variance(data: List[float]) -> float:
    """Computes unbiased sample variance (Bessel's correction)."""
    if not data:
        raise ValueError("Data list cannot be empty.")
    
    mean = sum(data) / len(data)
    # BUG: divides by len(data) instead of len(data) - 1
    return sum((x - mean) ** 2 for x in data) / len(data)
