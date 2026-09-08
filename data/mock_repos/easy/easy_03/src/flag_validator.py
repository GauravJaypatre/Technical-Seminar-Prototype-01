from typing import Optional, Set

def validate_flag(flag: Optional[str], allowed_flags: Set[str]) -> bool:
    """Validates that a CLI flag is within the allowed set (case-insensitive)."""
    # BUG: does not check if flag is None before calling .lower()
    normalized = flag.lower()
    normalized_allowed = {f.lower() for f in allowed_flags}
    return normalized in normalized_allowed
