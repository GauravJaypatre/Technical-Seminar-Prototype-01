# Issue: AttributeError on NoneType optional CLI argument

**Task ID:** `easy_03`  
**Tier:** `easy`  
**Domain:** `CLI / System`  

## Description
`validate_flag(flag, allowed_flags)` raises `AttributeError: 'NoneType' object has no attribute 'lower'` when optional flag is passed as None. It should return False instead of crashing.
