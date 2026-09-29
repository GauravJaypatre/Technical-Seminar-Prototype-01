"""Unit tests verifying billing settlement and discount invoice calculations."""

from billing.models import CustomerAccount, InvoiceLineItem
from billing.service import CheckoutService


def test_standard_tax_exempt_checkout():
    """Exempt checkout without fractional cents."""
    service = CheckoutService(state="CA")
    account = CustomerAccount(account_id="ACC-1", email="corp@example.com", tax_exempt=True)
    items = [
        InvoiceLineItem(sku="SKU-1", description="Widget A", quantity=2, unit_price=10.0),
        InvoiceLineItem(sku="SKU-2", description="Widget B", quantity=1, unit_price=25.0),
    ]
    invoice = service.generate_invoice("INV-001", account, items)
    assert invoice.net_total == 45.0
    assert invoice.tax_amount == 0.0


def test_invoice_net_settlement_with_fractional_discount():
    """Reproduces commercial billing rounding defect.
    
    Subtotal: 3 items @ $31.50 with a $1.255 volume discount = $93.245.
    CA Tax (8.25%): 93.245 * 0.0825 = 7.6927125.
    Total before rounding: 93.245 + 7.6927125 = 100.9377125.
    With standard commercial rounding (round to nearest cent / half-up):
    100.9377125 rounds to $100.94.
    
    BUG: Truncation in core.currency.RoundingPolicy truncates to $100.93,
    causing an under-billing discrepancy!
    """
    service = CheckoutService(state="CA")
    account = CustomerAccount(account_id="ACC-2", email="customer@example.com", tax_exempt=False)
    items = [
        InvoiceLineItem(
            sku="PROD-HIGH",
            description="Enterprise Cloud Node",
            quantity=3,
            unit_price=31.50,
            discount_amount=1.255,  # Subtotal = 93.245
        ),
    ]
    invoice = service.generate_invoice("INV-002", account, items)
    # Expected standard financial rounded settlement: 100.94
    assert invoice.net_total == 100.94, f"Expected 100.94, got {invoice.net_total}"
