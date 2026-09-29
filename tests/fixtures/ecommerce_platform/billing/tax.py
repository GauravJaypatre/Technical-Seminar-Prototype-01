"""Jurisdiction tax calculations and state exemption policies."""

from __future__ import annotations


class TaxCalculator:
    """Calculates applicable sales taxes per geographic jurisdiction."""

    RATES: dict[str, float] = {
        "CA": 0.0825,
        "NY": 0.08875,
        "TX": 0.0625,
        "WA": 0.065,
    }

    @classmethod
    def get_rate_for_state(cls, state: str) -> float:
        return cls.RATES.get(state.upper(), 0.05)

    @classmethod
    def compute_tax(cls, subtotal: float, state: str, is_exempt: bool = False) -> float:
        if is_exempt:
            return 0.0
        rate = cls.get_rate_for_state(state)
        return subtotal * rate
