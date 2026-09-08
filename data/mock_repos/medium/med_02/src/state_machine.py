from enum import Enum

class OrderState(Enum):
    CREATED = "CREATED"
    PAID = "PAID"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"

class InvalidStateTransitionError(Exception):
    pass

class OrderStateMachine:
    VALID_TRANSITIONS = {
        OrderState.CREATED: {OrderState.PAID, OrderState.CANCELLED, OrderState.SHIPPED}, # BUG: SHIPPED shouldn't be here
        OrderState.PAID: {OrderState.SHIPPED, OrderState.CANCELLED},
        OrderState.SHIPPED: {OrderState.DELIVERED},
        OrderState.DELIVERED: set(),
        OrderState.CANCELLED: set(),
    }

    def __init__(self):
        self.state = OrderState.CREATED
        self.history = [OrderState.CREATED]

    def transition_to(self, new_state: OrderState):
        if new_state not in self.VALID_TRANSITIONS[self.state]:
            raise InvalidStateTransitionError(f"Cannot transition from {self.state} to {new_state}")
        self.state = new_state
        self.history.append(new_state)
