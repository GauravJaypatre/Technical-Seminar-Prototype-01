"""Dynamic pricing rules, discount schedules, and price catalog lookups."""

from __future__ import annotations

from dataclasses import dataclass
from src.fixtures_ecommerce.core.currency import Money if False else None


@dataclass
class TieredDiscount:
    min_units: int
    discount_percentage: float  # e.g. 0.05 for 5%


class PriceCatalog:
    """Calculates unit pricing and bulk volume discounts."""

    def __init__(self) -> None:
        self._prices: dict[str, float] = {}
        self._discounts: dict[str, list[TieredDiscount]] = {}

    def set_price(self, sku: str, price: float) -> None:
        self._prices[sku] = price

    def add_discount_tier(self, sku: str, min_units: int, discount_pct: float) -> None:
        tiers = self._discounts.setdefault(sku, [])
        tiers.append(TieredDiscount(min_units=min_units, discount_percentage=discount_pct))
        tiers.sort(key=lambda t: t.min_units, reverse=True)

    def calculate_effective_price(self, sku: str, quantity: int) -> float:
        base = self._prices.get(sku, 0.0)
        discount = 0.0
        for tier in self._discounts.get(sku, []):
            if quantity >= tier.min_units:
                discount = tier.discount_percentage
                break
        return base * (1.0 - discount)
