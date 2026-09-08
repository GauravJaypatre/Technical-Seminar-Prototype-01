# Issue: Shared mutable default causes plugin state cross-contamination

**Task ID:** `hard_06`  
**Tier:** `hard`  
**Domain:** `Extensibility / Architecture`  

## Description
`PluginContext` across `plugin.py` and `registry.py` uses a class-level mutable dictionary `metadata={}` as a default argument. When one plugin injects configuration or metadata into its execution context, it leaks into all other concurrently executed plugins.
