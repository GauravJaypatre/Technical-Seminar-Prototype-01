def merge_configs(base: dict, override: dict) -> dict:
    """Merges override dictionary into base dictionary recursively."""
    result = base.copy()
    # BUG: shallow update completely replaces nested dictionaries
    result.update(override)
    return result
