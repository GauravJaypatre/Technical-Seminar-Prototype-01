"""Billing data structures, invoice line items, and account profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class InvoiceLineItem:
    sku: str
    description: str
    quantity: int
    unit_price: float
    discount_amount: float = 0.0

    @property
    def raw_subtotal(self) -> float:
        return (self.quantity * self.unit_price) - self.discount_amount


@dataclass
class CustomerAccount:
    account_id: str
    email: str
    tax_exempt: bool = False


@dataclass
class BillingInvoice:
    invoice_id: str
    account_id: str
    items: List[InvoiceLineItem] = field(default_factory=list)
    tax_amount: float = 0.0
    net_total: float = 0.0
    paid: bool = False
