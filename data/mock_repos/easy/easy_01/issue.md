# Issue: Query parser drops keys with empty values

**Task ID:** `easy_01`  
**Tier:** `easy`  
**Domain:** `Networking / Web`  

## Description
When parsing query strings like `?key=&other=val`, `parse_query_string()` completely ignores keys with empty values (`key`). According to standard URL query specification, keys with empty values should be retained with an empty string value `''` instead of being dropped.
