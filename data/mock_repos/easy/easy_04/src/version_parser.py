import re

def tokenize_version(version_string: str) -> list[int]:
    """Splits semantic version string into a list of integer components."""
    if not version_string:
        return []
    # BUG: '.' is not escaped in regex pattern
    parts = re.split(r'.', version_string)
    parts = [p for p in parts if p]
    result = []
    for p in parts:
        if p.isdigit():
            result.append(int(p))
    return result
