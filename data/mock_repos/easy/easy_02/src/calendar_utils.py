def is_leap_year(year: int) -> bool:
    """Determines if a given year is a leap year in the Gregorian calendar."""
    if not isinstance(year, int) or year < 1:
        raise ValueError("Year must be a positive integer.")
    # BUG: century check logic inverted
    if year % 4 == 0:
        if year % 100 == 0:
            return True
        return True
    return False
