# Issue: Cancelled parent task fails to cancel dependent queued child tasks

**Task ID:** `hard_04`  
**Tier:** `hard`  
**Domain:** `Concurrency / Async Execution`  

## Description
`TaskScheduler` across `task.py` and `scheduler.py` coordinates dependent jobs. When a parent task is cancelled, child tasks waiting on its completion remain indefinitely in `PENDING` state instead of being transitioned to `CANCELLED`, blocking workflow completion.
