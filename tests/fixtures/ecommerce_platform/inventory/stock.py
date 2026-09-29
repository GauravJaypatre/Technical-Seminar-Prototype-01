"""Inventory stock management, warehouse levels, and allocation tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StockItem:
    sku: str
    name: str
    available_quantity: int
    reserved_quantity: int = 0


class StockManager:
    """Manages multi-warehouse catalog items and availability thresholds."""

    def __init__(self) -> None:
        self._inventory: dict[str, StockItem] = {}

    def register_item(self, sku: str, name: str, quantity: int) -> StockItem:
        item = StockItem(sku=sku, name=name, available_quantity=quantity)
        self._inventory[sku] = item
        return item

    def get_item(self, sku: str) -> StockItem | None:
        return self._inventory.get(sku)

    def deduct_stock(self, sku: str, quantity: int) -> bool:
        item = self._inventory.get(sku)
        if not item or item.available_quantity < quantity:
            return False
        item.available_quantity -= quantity
        return True
