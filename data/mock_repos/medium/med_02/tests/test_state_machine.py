import pytest
from src.state_machine import OrderStateMachine, OrderState, InvalidStateTransitionError

def test_valid_order_flow():
    sm = OrderStateMachine()
    sm.transition_to(OrderState.PAID)
    sm.transition_to(OrderState.SHIPPED)
    sm.transition_to(OrderState.DELIVERED)
    assert sm.state == OrderState.DELIVERED

def test_invalid_skip_paid_transition():
    # FAILS ON UNPATCHED CODE
    sm = OrderStateMachine()
    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(OrderState.SHIPPED)

def test_cannot_transition_from_cancelled():
    sm = OrderStateMachine()
    sm.transition_to(OrderState.CANCELLED)
    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(OrderState.PAID)
