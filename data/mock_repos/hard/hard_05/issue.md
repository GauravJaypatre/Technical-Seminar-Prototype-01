# Issue: Query builder renders raw table name instead of alias in WHERE clause

**Task ID:** `hard_05`  
**Tier:** `hard`  
**Domain:** `Database / ORM`  

## Description
The query generator across `models.py` and `builder.py` allows aliasing joined tables (e.g., `users AS u`). However, `where()` conditions construct column identifiers using the original table name rather than the alias when an alias is present, generating invalid SQL syntax.
