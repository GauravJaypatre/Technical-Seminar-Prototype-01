# Issue: Failing subscriber stops execution of remaining event listeners

**Task ID:** `med_06`  
**Tier:** `medium`  
**Domain:** `Event-Driven / Architecture`  

## Description
`EventDispatcher.dispatch()` immediately bubbles up exceptions if one subscriber fails, preventing all subsequent registered listeners from receiving the event. It should collect exceptions into a composite `DispatchAggregateError` and guarantee all listeners are called.
