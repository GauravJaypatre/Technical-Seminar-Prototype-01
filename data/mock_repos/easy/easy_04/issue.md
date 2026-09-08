# Issue: Unescaped dot in version splitter splits on any character

**Task ID:** `easy_04`  
**Tier:** `easy`  
**Domain:** `Parsing / Packaging`  

## Description
`tokenize_version(v_str)` uses `re.split('.', v_str)` with an unescaped dot, which matches any character in regex, causing strings to split improperly on non-dot characters.
