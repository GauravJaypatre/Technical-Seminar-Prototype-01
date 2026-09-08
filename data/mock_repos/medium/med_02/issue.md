# Issue: Order state machine allows invalid transition skipping payment

**Task ID:** `med_02`  
**Tier:** `medium`  
**Domain:** `Business Logic / State Machine`  

## Description
`OrderStateMachine` allows transitioning directly from `CREATED` to `SHIPPED`, bypassing `PAID`. Transitions must adhere strictly to valid sequences: CREATED -> PAID -> SHIPPED -> DELIVERED, or CANCELLED from CREATED/PAID. Any invalid transition must raise `InvalidStateTransitionError`.
