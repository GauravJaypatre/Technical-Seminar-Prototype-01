# Issue: Key existence check reports expired keys as present

**Task ID:** `hard_02`  
**Tier:** `hard`  
**Domain:** `Caching / Data Structures`  

## Description
`TTLLRUCache` across `storage.py` and `cache_manager.py` maintains an index dictionary and linked nodes. When checking `contains_key()`, it checks the index dictionary without validating if the key has expired against `ttl`, returning True for expired keys.
