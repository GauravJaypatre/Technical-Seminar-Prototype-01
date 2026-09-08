# Issue: Shallow dict merge overwrites nested settings

**Task ID:** `easy_05`  
**Tier:** `easy`  
**Domain:** `Configuration / Utils`  

## Description
`merge_configs(base, override)` uses shallow `.update()`, which wipes out all unmentioned nested keys in sub-dictionaries instead of recursively merging them.
