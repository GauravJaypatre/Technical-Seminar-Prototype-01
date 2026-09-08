# Issue: Header dict lookup fails on case variations

**Task ID:** `med_01`  
**Tier:** `medium`  
**Domain:** `Networking / HTTP`  

## Description
`CaseInsensitiveDict` normalizes keys during `__setitem__` but fails to normalize in `get()`, `__contains__`, and `pop()`, causing lookups like `'content-type' in headers` to return False when stored as `'Content-Type'`.
