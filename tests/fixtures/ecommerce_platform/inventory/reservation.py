"""Temporary cart hold reservations and expiration policies."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from typing import Optional


@dataclass
class ItemReservation:
    reservation_id: str
    cart_id: str
    sku: str
    quantity: int
    expires_at: datetime.datetime


class ReservationManager:
    """Manages lock holds during customer checkout flows."""

    def __init__(self, ttl_minutes: int = 15) -> None:
        self.ttl = datetime.timedelta(minutes=ttl_minutes)
        self._reservations: dict[str, ItemReservation] = {}

    def reserve(self, reservation_id: str, cart_id: str, sku: str, quantity: int) -> ItemReservation:
        res = ItemReservation(
            reservation_id=reservation_id,
            cart_id=cart_id,
            sku=sku,
            quantity=quantity,
            expires_at=datetime.datetime.utcnow() + self.ttl,
        )
        self._reservations[reservation_id] = res
        return res

    def get_reservation(self, reservation_id: str) -> Optional[ItemReservation]:
        res = self._reservations.get(reservation_id)
        if res and res.expires_at > datetime.datetime.utcnow():
            return res
        return None
