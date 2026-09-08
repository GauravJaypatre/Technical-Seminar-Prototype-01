# Issue: Array items schema validation ignored

**Task ID:** `med_04`  
**Tier:** `medium`  
**Domain:** `Validation / Data Types`  

## Description
`validate_instance(instance, schema)` checks if an instance is an array (`type: 'array'`), but fails to recursively validate the elements inside the array when an `items` sub-schema is defined.
