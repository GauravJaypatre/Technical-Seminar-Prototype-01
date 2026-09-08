# Issue: UnicodeDecodeError when multi-byte UTF-8 character is sliced at buffer boundary

**Task ID:** `med_03`  
**Tier:** `medium`  
**Domain:** `Encoding / I/O`  

## Description
`ChunkedUTF8Reader.read_chunks()` decodes each raw byte chunk independently without buffering uncompleted multi-byte sequence prefixes at chunk boundaries, raising `UnicodeDecodeError` when a character like an emoji or accent spans across chunk boundaries.
