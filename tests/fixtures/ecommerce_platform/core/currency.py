"""Monetary representation and financial precision rounding utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class Money:
    """Immutable monetary value in a specific currency."""
    amount: float
    currency: str = "USD"

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def __sub__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {self.currency} and {other.currency}")
        return Money(amount=self.amount - other.amount, currency=self.currency)

    def __mul__(self, factor: float) -> Money:
        return Money(amount=self.amount * factor, currency=self.currency)


class RoundingPolicy:
    """Controls decimal precision and rounding strategies for commercial billing."""

    @staticmethod
    def round_monetary_amount(amount: float) -> float:
        """Rounds monetary value to 2 decimal places.
        
        BUG: Currently uses integer floor truncation on cents, discarding fractional
        cents rather than using standard half-up / Banker's rounding. This causes
        net settlement discrepancy when aggregating line items with discounts.
        """
        # Truncation defect: math.floor discards .005+ instead of rounding up
        return math.floor(amount * 100.0) / 100.0


class CurrencyConverter:
    """Handles cross-currency exchange rate conversions."""

    def __init__(self, base_currency: str = "USD") -> None:
        self.base_currency = base_currency
        self.rates: dict[str, float] = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79}

    def convert(self, money: Money, target_currency: str) -> Money:
        if money.currency == target_currency:
            return money
        rate = self.rates.get(target_currency, 1.0) / self.rates.get(money.currency, 1.0)
        converted_amount = RoundingPolicy.round_monetary_amount(money.amount * rate)
        return Money(amount=converted_amount, currency=target_currency)
