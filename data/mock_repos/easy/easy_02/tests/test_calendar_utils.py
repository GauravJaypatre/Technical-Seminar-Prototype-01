import pytest
from src.calendar_utils import is_leap_year

def test_regular_leap_year():
    assert is_leap_year(2024) is True
    assert is_leap_year(1996) is True

def test_non_leap_year():
    assert is_leap_year(2023) is False
    assert is_leap_year(2019) is False

def test_century_non_leap_years():
    # FAILS ON UNPATCHED CODE
    assert is_leap_year(1900) is False
    assert is_leap_year(2100) is False
    assert is_leap_year(1800) is False

def test_quad_century_leap_years():
    assert is_leap_year(2000) is True
    assert is_leap_year(1600) is True
