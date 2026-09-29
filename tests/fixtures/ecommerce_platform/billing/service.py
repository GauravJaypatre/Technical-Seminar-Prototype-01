"""Billing checkout service coordinating line item settlement and invoices."""

from __future__ import annotations

from typing import List
from core.currency import RoundingPolicy
from billing.models import BillingInvoice, CustomerAccount, InvoiceLineItem
from billing.tax import TaxCalculator


class CheckoutService:
    """Processes carts into settled invoices with applicable discounts and taxes."""

    def __init__(self, state: str = "CA") -> None:
        self.state = state

    def generate_invoice(
        self,
        invoice_id: str,
        account: CustomerAccount,
        items: List[InvoiceLineItem],
    ) -> BillingInvoice:
        """Calculates line item totals, applies tax, and establishes net total.
        
        The net settlement relies on core.currency.RoundingPolicy.round_monetary_amount.
        When items have fractional cents from volume discounts, integer truncation
        in RoundingPolicy causes a 1-cent discrepancy under-billing customers.
        """
        raw_subtotal = 0.0
        for item in items:
            raw_subtotal += item.raw_subtotal

        # Apply state jurisdiction tax
        tax = TaxCalculator.compute_tax(raw_subtotal, self.state, is_exempt=account.tax_exempt)

        # Net total calculation uses RoundingPolicy
        net_raw = raw_subtotal + tax
        net_settled = RoundingPolicy.round_monetary_amount(net_raw)

        return BillingInvoice(
            invoice_id=invoice_id,
            account_id=account.account_id,
            items=items,
            tax_amount=RoundingPolicy.round_monetary_amount(tax),
            net_total=net_settled,
            paid=False,
        )
