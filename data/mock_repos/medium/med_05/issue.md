# Issue: Rate limiter boundary allows request exceeding window limit

**Task ID:** `med_05`  
**Tier:** `medium`  
**Domain:** `Concurrency / Algorithms`  

## Description
`SlidingWindowRateLimiter` uses strict `<` comparison for timestamp eviction instead of `<=`, allowing requests on the exact window boundary to be double-counted or retaining expired requests.
