# Issue: Nested comprehension variable leaks into enclosing function scope

**Task ID:** `hard_03`  
**Tier:** `hard`  
**Domain:** `Compilers / Static Analysis`  

## Description
The AST variable analyzer across `scope.py` and `analyzer.py` models symbol tables. When encountering a list comprehension or lambda, it reuses the active function scope instead of pushing a child lexical scope, causing loop target variables inside comprehensions to overwrite identically named outer variables.
